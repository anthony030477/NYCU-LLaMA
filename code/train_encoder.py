import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn
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
    for ids_a, ids_b, input_masks,_ in (bar:=tqdm(train_dataloader,ncols=0)):
        loss=0
        bs=ids_a.shape[0]
        optimizer.zero_grad()
        counter+=1
        ids_a=ids_a.to(device)
        ids_b=ids_b.to(device)
        input_masks=input_masks.to(device)


        z_on, q_on = model_on(ids_a,input_masks)
        with torch.no_grad():
            z_off, q_off = model_off(ids_b,input_masks)
        # del z_on, q_off     #delete online latent and offline prediction

        loss += 2-2*((q_on*z_off).sum(dim=1)/(torch.norm(q_on, dim=1)*torch.norm(z_off, dim=1))).mean()
        # MSE of online prediction and offline latent
        # q_on=F.normalize(q_on)
        # z_off=F.normalize(z_off)
        # loss = torch.mean(( q_on - z_off )**2)

        z_on, q_on = model_on(ids_b,input_masks)
        with torch.no_grad():
            z_off, q_off = model_off(ids_a,input_masks)
        # del z_on, q_off     #delete online latent and offline prediction

        loss += 2-2*((q_on*z_off).sum(dim=1)/(torch.norm(q_on, dim=1)*torch.norm(z_off, dim=1))).mean()
        # MSE of online prediction and offline latent
        # q_on=F.normalize(q_on)
        # z_off=F.normalize(z_off)
        # loss += torch.mean(( q_on - z_off )**2)


        loss.backward()
        optimizer.step()

        if counter%update_step==0:
            momentum_update(model_on, model_off)


        # calculate similarity of each sample
        with torch.no_grad():
            sim=(z_on@z_on.T)/(torch.norm(z_on,dim=1)@torch.norm(z_on,dim=1).T)

        all_sim=0
        for i in range(bs):
            self_sim = sim[i, i]
            other = torch.cat([sim[i, :i], sim[i,i:]], dim=0).mean()
            all_sim += self_sim/other
        all_sim/=bs
        # log
        if losses==0:
            losses = loss.item()
        losses=0.96*losses+0.04*loss.item()
        bar.set_description(f'epoch[{epoch+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss {losses:.4f}, self sim:{all_sim:.4f}')
    lr_scher.step(losses)

if __name__=='__main__':
    dataset=trainDataset()

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True,collate_fn=collect_fn(drop=0.2))
    model_on=Roberta()
    model_off=Roberta()
    model_on.to(device)
    model_off.to(device)

    model_on.load_state_dict(torch.load('save/save_099.pt', 'cpu'))

    momentum_update(model_on, model_off, 1) #full copy online to offline
    optimizer = torch.optim.AdamW(model_on.parameters(), lr=5e-5, weight_decay=1e-2)
    num_epochs=1000
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, cooldown=20, min_lr=1e-6)

    for i in range(num_epochs):
        trainer(i, model_on, model_off)
        if (i+1)%10==0:
            torch.save(model_on.state_dict(),f'save/save_{i+1:03d}.pt')
    torch.save(model_on.state_dict(),'save/save.pt')
