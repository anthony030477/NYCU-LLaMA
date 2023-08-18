import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn,trainDataset_NEWS,collect_fn_news
from model import Roberta
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mse_loss=nn.MSELoss()

lam=25
mu=25
nu=1


def trainer(i):
    loss_list=[]
    model.train()
    for input_ids,input_masks ,labels,_ in (bar:=tqdm(train_dataloader,ncols=0)):
        input_ids=input_ids.to(device)
        labels=labels.to(device)
        input_masks=input_masks.to(device)

        optimizer.zero_grad()
        _,z_a  = model(input_ids,input_masks)
        _,z_b =model(input_ids,input_masks)
        sim_loss = mse_loss(z_a, z_b)
        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        N=input_ids.shape[0]
        D=z_a.shape[1]
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = torch.masked_select(cov_z_a, (1-torch.eye(cov_z_a.shape[0],device=device)).to(torch.bool)).pow_(2).sum() / D
        + torch.masked_select(cov_z_b, (1-torch.eye(cov_z_b.shape[0],device=device)).to(torch.bool)).pow_(2).sum() / D
        # loss
        loss = lam * sim_loss + mu * std_loss + nu * cov_loss

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # print(labels.shape)
        acc=0#torch.sum(torch.argmax(predict,dim=-1)==labels)/N
        
        bar.set_description(f'epoch[{i+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss {sum(loss_list)/len(loss_list):.4f} acc {acc:.4f}')
    lr_scher.step()

if __name__=='__main__':
    dataset=trainDataset_NEWS()
    
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=collect_fn_news)
    model=Roberta()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6/(256/32)**0.5,weight_decay=1e-6) 
    num_epochs=1000
    lr_scher=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-8/(256/32)**0.5, )
    
    for i in range(num_epochs):
        trainer(i)
    torch.save(model.state_dict(),'save/save.pt')