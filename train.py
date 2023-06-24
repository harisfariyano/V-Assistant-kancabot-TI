import numpy as np #library python yang digunakan untuk bekerja dengan array dan juga memiliki fungsi yang bekerja dalam domain aljabar linier, transformasi fourier, dan matriks
import random #untuk menghasilkan angka acak di Python.
import json # gunakan modul json

import torch # library tensor deep learning yang dioptimalkan berdasarkan Python dan Torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
# buka file JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop melalui setiap kalimat intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # tambahkan ke  tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tandai setiap kata dalam kalimat
        w = tokenize(pattern)
        # tambahkan ke daftar kata 
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word (menghilangkan imbuhan pada suatu kata.)
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort(hapus duplikat dan diurutkan)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Parameter hiper 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # mendukung pengindeksan sehingga kumpulan data[i] dapat digunakan untuk mendapatkan sampel ke-i
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # kita dapat memanggil len(dataset) untuk mengembalikan ukurannya
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Latih modelnya
#Epoch merupakan hyperparameter yang menentukan berapa kali algoritma deep learning 
# bekerja melewati seluruh dataset baik secara forward maupun backward.
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# cetak isi data final loss
print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
#cetak isi data training complete menyimpan di sebuah file di data.pth
print(f'training complete. file saved to {FILE}')
