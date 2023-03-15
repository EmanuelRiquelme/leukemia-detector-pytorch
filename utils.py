import torch
import os 

def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/model.pt'))

def save_model(model):
    torch.save(model.state_dict(), f'{os.getcwd()}/model.pt')

def validation(data,model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    it = iter(data)
    acc = []
    with torch.no_grad():
        for _ in range(len(data)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            pred = model(input)
            pred = torch.argmax(pred,-1)
            acc.append(((pred == target.flatten()).nonzero()).size(0)/target.size(0))
    model.train()
    return sum(acc)/len(acc)
