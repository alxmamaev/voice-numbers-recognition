import os
import torch
import numpy as np
from torch import nn
from model.asr import ASR
from model.loss import ASRLoss
from data.dataset import get_tain_val_loader
from argparse import ArgumentParser
from ignite.engine import Engine, create_supervised_trainer, Events
from torch.utils.tensorboard import SummaryWriter



def parse():
    parser = ArgumentParser()
    parser.add_argument("--datapath", default="dataset")
    parser.add_argument("--train_meta", default="dataset/train.csv")
    parser.add_argument("--val_meta", default="dataset/val.csv")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    
    return parser.parse_args()


def main(args):
    model = ASR()
    criterion = ASRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    writer = SummaryWriter()
    
    os.makedirs("checkpoints/", exist_ok=True)
    
    train_loader, val_loader = get_tain_val_loader(args.datapath, args.train_meta, args.val_meta, args.batch_size)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        model.train()
        writer.add_scalar("loss", trainer.state.output, trainer.state.iteration)
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        model.eval()
        predictions = []
        groun_truth = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch
                features = features.to(device)
                pred = model(features).cpu()
                pred = torch.argmax(pred, dim=2)
                
                predictions.append(pred)
                groun_truth.append(labels)
        
        predictions = torch.cat(predictions)
        groun_truth = torch.cat(groun_truth)
        
        output = groun_truth == predictions
        
        accuracy = []
        for i in range(6):
            acc = output[:,i].float().mean()
            writer.add_scalar("accuracy_{}".format(i), acc, trainer.state.epoch)
            accuracy.append(acc)
        
        writer.add_scalar("accuracy", np.mean(accuracy), trainer.state.epoch)
        
        state_dict = model.state_dict()
        torch.save(state_dict, "checkpoints/checkpoint_epoch_{}.pth".format(trainer.state.epoch))
        
                
        
    trainer.run(train_loader, max_epochs=args.epochs)
    writer.close()


if __name__ == "__main__":
    args = parse()
    main(args)