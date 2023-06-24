import torch #librray  tensor deep learning yang dioptimalkan berdasarkan Python dan Torch
import torch.nn as nn

#membuat class neural
#Setelah mendapat nilai biner dari teks input, maka dimasukan nilai biner tersebut 
# ke dalam input layer yang nantinya akan
#memberi sinyal kepada hidden layer dari
#arsitektur neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
