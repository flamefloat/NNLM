import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
print(word_list)
word_list = list(set(word_list))
print(word_list)
word_dict = {w: i for i, w in enumerate(word_list)}
print(word_dict)
number_dict = {i: w for i, w in enumerate(word_list)}
print(number_dict)
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))
print(input_batch, target_batch)

input_size = n_step * m
hidden_size = n_hidden

class NNLM(nn.Module):
    def __init__(self, input_size, hidden_size, n_class):
        super(NNLM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.emb = nn.Embedding(n_class, m)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.n_class)
    def forward(self, x):
        x = self.emb(x)
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = torch.tanh(x)
        output = self.fc2(x)
        return output


model = NNLM(input_size, hidden_size, n_class)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5000):
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

print(predict)
# Test
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()]) 
#squeeze 去掉维度值为1的维度 item:返回张量的值