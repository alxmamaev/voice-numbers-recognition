import torch
from torch import nn
from model.modules import VGGExtractor, RNNLayer, Attention

class ASR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vgg = VGGExtractor(40)
        self.rnn = RNNLayer()
        self.attention = Attention()
        self.fc = nn.Linear(512, 10)
        
        
    def forward(self, x):
        output = self.vgg(x, x.shape[1])[0]
        output = self.rnn(output)
        output = self.attention(output)
        output = self.fc(output)
        
        return output