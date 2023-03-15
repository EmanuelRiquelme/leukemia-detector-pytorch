import torch
import os
from torch.utils.data import DataLoader
from model import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from dataset import train_cancer,test_cancer
from utils import validation,save_model,load_model
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
train_data = train_cancer()
val_data = test_cancer()
batch_size = 24
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4)
val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True,num_workers = 4)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Model().to(device)
opt = optim.Adam(model.parameters(),lr=0.0005)
loss_fn = nn.CrossEntropyLoss()
epochs = 15

def train_pipeline(train_data = train_dataloader,model = model,
                    loss_fn = loss_fn,opt = opt,epochs = epochs,device = device,val_data = val_data,threshold = .85):
    for epoch in  (t := trange(epochs)):
        it = iter(train_data)
        for _ in range(len(train_data)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output,target)
            loss.backward()
            opt.step()
        temp_val = validation(val_data,model)
        writer.add_scalar('Loss',temp_val,epoch)
        t.set_description(f'validation: {temp_val:.2f}')
        writer.add_scalar('Loss',temp_val,epoch)
        if temp_val >= threshold:
            break

if __name__ == '__main__':
    print(f'initial validation: {validation(val_data,model):.2f}')
    train_pipeline()
    save_model(model)
