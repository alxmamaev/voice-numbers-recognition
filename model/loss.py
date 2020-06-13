import torch
from torch import nn 


class ASRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, pred, labels):
        loss = torch.zeros(6)
        
        for i in range(6):
            loss[i] = self.criterion(pred[:, i, :], labels[:, i])
            
        return torch.mean(loss)