
import numpy as np
import pandas as pd

# 读取数据
# 数据来自 https://www.kaggle.com/crowdflower/twitter-airline-sentiment
path =  '/home/bb5/xw/pytourch_demo/Tweets.csv'
data = pd.read_csv(path)
data.head()
reviews = np.array(data['text'])[:14000]
labels = np.array(data['airline_sentiment'])[:14000]

# 计算labels数量
from collections import Counter
print('labels : ', Counter(labels))

# text 文本处理 提交标点符号等等无意义的字符
punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
all_reviews = 'separator'.join(reviews)
all_reviews = all_reviews.lower()
all_text = ''.join([c for c in all_reviews if c not in punctuation])

# 分割
reviews_split = all_text.split('separator')
all_text = ' '.join(reviews_split)

# 提取所有单词
words = all_text.split()


# 每句话的单词,且不含'@' http
new_reviews = []
for review in reviews_split:
    review = review.split()
    new_text = []
    for word in review:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    new_reviews.append(new_text)



## 单词字典 映射关系
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


# 评论量化
reviews_ints = []
for review in new_reviews:
    reviews_ints.append([vocab_to_int[word] for word in review])

# 词汇统计
tmp = [ len(i) for i in reviews_ints]
print('单词唯一数 ', len((vocab_to_int)))  # should ~ 74000+
print('评论最大长度 ',max(tmp))


# 2=positive, 1=neutral, 0=negative label conversion
encoded_labels = []
for label in labels:
    if label == 'neutral':
        encoded_labels.append(1)
    elif label == 'negative':
        encoded_labels.append(0)
    else:
        encoded_labels.append(2)

encoded_labels = np.asarray(encoded_labels)



def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

# 每个评论前30个字符
seq_length = 30
features = pad_features(reviews_ints, seq_length=seq_length)


split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of the resultant feature data
print("Feature Shapes:")
print("\nTrain  : {}".format(train_x.shape),
      "\nValidation  : {}".format(val_x.shape),
      "\nTest  : {}".format(test_x.shape))




import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')





import torch.nn as nn
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size ):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 200)
        self.lstm = nn.LSTM(200, 128, 2, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(3840, 640)
        self.fc3 = nn.Linear(640, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)


    def forward(self, x,hidden):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds,hidden)
        out = out.reshape(50,-1)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.dropout(out)
        out = self.fc5(out)

        return out, hidden

    def init_hidden(self, ):

        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(2, 50, 128).zero_().cuda(),
                      weight.new(2, 50, 128).zero_().cuda())
        else:
            hidden = (weight.new(2, 50, 128).zero_(),
                      weight.new(2, 50, 128).zero_())

        return hidden



# 参数设定
vocab_size = len(vocab_to_int)+1
net = SentimentRNN(vocab_size,)


# loss optimizer
lr=0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)




epochs = 50
counter = 0
clip=5 # gradient clipping

if(train_on_gpu):net.cuda()
net.train()
for e in range(epochs):
    h = net.init_hidden()

    for inputs, labels in train_loader:
        counter += 1
        if(train_on_gpu):inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        net.zero_grad()
        output, h = net(inputs,h)
        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()


        if counter % 100 == 0:
            val_h = net.init_hidden()
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                if(train_on_gpu):inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs,val_h)
                val_loss = criterion(output, labels)
                val_losses.append(val_loss.item())

            net.train()
            print(
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


def val(test_loader):
    test_losses = []
    num_correct = 0
    h = net.init_hidden()

    net.eval()
    for inputs, labels in test_loader:
        if (train_on_gpu): inputs, labels = inputs.cuda(), labels.cuda()
        output, h = net(inputs, h)

        #  loss
        test_loss = criterion(output, labels)
        test_losses.append(test_loss.item())

        # accuracy
        _, pred = torch.max(output.data, 1)
        num_correct += torch.sum(labels == pred).double()

    print(" loss: {}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader.dataset)
    print(" accuracy: {}".format(test_acc))




print('train ')
val(train_loader)
print('valid ')
val(valid_loader)
print('test ')
val(test_loader)




