import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn,trainDataset_NEWS,collect_fn_news
from model import Roberta, momentum_update
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mse_loss=nn.MSELoss()



update_step=1


def trainer(epoch, model_on:nn.Module, model_off:nn.Module):
    losses = 0
    model_on.train()
    model_off.eval()
    counter=0
    for input_ids,input_masks ,labels,_ in (bar:=tqdm(train_dataloader,ncols=0)):
        counter+=1
        input_ids=input_ids.to(device)
        labels=labels.to(device)
        input_masks=input_masks.to(device)

        optimizer.zero_grad()
        z_on, q_on = model_on(input_ids,input_masks)
        with torch.no_grad():
            z_off, q_off = model_off(input_ids,input_masks)
        del z_on, q_off     #delete online latent and offline prediction


        # MSE of online prediction and offline latent
        loss = torch.mean(( q_on - z_off )**2)

        loss.backward()
        optimizer.step()

        if counter%update_step==0:
            momentum_update(model_on, model_off)

        # log
        if losses==0:
            losses = loss.item()
        losses=0.98*losses+0.02*loss.item()
        bar.set_description(f'epoch[{i+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss {losses:.4f}')
    lr_scher.step()

if __name__=='__main__':
    dataset=trainDataset_NEWS()

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=collect_fn_news)
    model_on=Roberta()
    model_off=Roberta()
    model_on.to(device)
    model_off.to(device)

    momentum_update(model_on, model_off, 1) #full copy online to offline
    optimizer = torch.optim.AdamW(model_on.parameters(), lr=3e-5, weight_decay=1e-2)
    num_epochs=1000
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, cooldown=20, min_lr=1e-6)

    for i in range(num_epochs):
        trainer(i)
    torch.save(model_on.state_dict(),'save/save.pt')
