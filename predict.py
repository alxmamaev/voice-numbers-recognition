import os
from argparse import ArgumentParser
from data.feature_extractor import ExtractAudioFeature
import torch
from model.asr import ASR
import pandas as pd
from tqdm import tqdm


def parse():
    parser = ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("checkpoint")
    
    return parser.parse_args()
    

def predict2number(pred):
    pred = torch.argmax(pred[0], dim=1)
    
    output = ""
    
    i = 0
    while pred[i] == 0 and i < len(pred):
        i += 1
    
    while i < len(pred):
        output += str(pred[i])
        i += 1
    
        
    if not output:
        return 0
    
    return int(output)



def main(args):
    print("Start predicting")
    model = ASR()
    feature_extractor = ExtractAudioFeature()
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
        
    def extract_features(path):
        features = feature_extractor(path)
        features = features[0].transpose(0, 1)
    
        pad_features = torch.ones(382, 40) * -15.9
        pad_features[:features.shape[0]] = features
        
        pad_features = pad_features / 16.0
        
        return pad_features
        
        
    model.eval()
    model = model.to(device)
    
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    
    input_data = pd.read_csv(args.input_csv)
    
    paths = input_data["path"].values.tolist()
    numbers = []
    
    with torch.no_grad():
        for path in tqdm(paths):
            path = os.path.join(args.datapath, path)
            features = extract_features(path)
            features = features.unsqueeze(0)
            features = features.to(device)
            
            pred = model(features).cpu()
            number = predict2number(pred)
            numbers.append(number)
            
            
    output_csv = pd.DataFrame()
    output_csv["path"] = paths
    output_csv["number"] = numbers
    
    output_csv.to_csv(args.output_csv)
    print("Ready")



if __name__ == "__main__":
    args = parse()
    main(args)