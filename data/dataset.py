import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from data.feature_extractor import ExtractAudioFeature



class AudioDataset(Dataset):
    def __init__(self, datapath, metadata, pad_size=382):
        data = pd.read_csv(metadata)
        
        self.pad_size = pad_size
        self.datapath = datapath
        self.files = data["path"].values.tolist()
        self.numbers = data["number"].astype(str).values.tolist()
        self.fe = ExtractAudioFeature()
        
        
    @staticmethod
    def __number_to_labels(number):
        labels = torch.zeros(6, dtype=torch.int64)
        
        for i, s in enumerate(number):
            labels[5 - i] = int(s) 
        
        return labels
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, indx):
        features = self.fe(os.path.join(self.datapath, 
                                        self.files[indx]))
        features = features[0].transpose(0, 1)
    
        pad_features = torch.ones(self.pad_size, 40) * -15.9
        pad_features[:features.shape[0]] = features
        
        pad_features = pad_features / 16.0
        
        
        labels = self.__number_to_labels(self.numbers[indx])
        
        return pad_features, labels
    
    
    
def get_tain_val_loader(datapath, train_meta, val_meta, batch_size):
    train_dataset = AudioDataset(datapath, train_meta)
    val_dataset = AudioDataset(datapath, val_meta)
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    return train_loader, val_loader