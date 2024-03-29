import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn
from model import Roberta, momentum_update,SBert
import torch.nn as nn
import torch.nn.functional as F
from utils import cos_sim, simCLR_loss, infonNCE_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mse_loss=nn.MSELoss()



update_step=1
class memory:
    def __init__(self, size=2048) -> None:
        self.size = size
        self.latents = None
    def get(self):
        return self.latents
    def add(self, latents:torch.Tensor):
        latents=latents.detach()
        if self.latents is None:
            self.latents=latents
        else:
            self.latents = torch.cat([latents, self.latents], dim=0)
            self.latents =  self.latents[:self.size]

def trainer(epoch, model_on:nn.Module, model_off:nn.Module):
    losses = 0
    model_on.train()
    model_off.eval()
    counter=0
    for ids_a, ids_b, mask_a, mask_b, text_a, taxt_b in (bar:=tqdm(train_dataloader,ncols=0)):

        loss=0
        bs=ids_a.shape[0]
        optimizer.zero_grad()
        counter+=1
        ids_a=ids_a.to(device)
        ids_b=ids_b.to(device)
        mask_a=mask_a.to(device)
        mask_b=mask_b.to(device)


        z_on = model_on(text_a)
        with torch.no_grad():
            z_off = model_off(taxt_b)
        bank.add(z_off)
        loss += infonNCE_loss(z_on, bank.get(), 0.01)

        z_on = model_on(taxt_b)
        with torch.no_grad():
            z_off = model_off(text_a)
        bank.add(z_off)
        loss += infonNCE_loss(z_on, bank.get(), 0.01)


        loss.backward()
        optimizer.step()

        if counter%update_step==0:
            momentum_update(model_on, model_off)


        # calculate similarity of each sample
        with torch.no_grad():
            sim=cos_sim(z_on,z_off)

        all_sim=0
        for i in range(bs):
            self_sim = sim[i, i]
            other = sim[i].mean()
            all_sim += self_sim-other
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

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True,collate_fn=collect_fn(drop=0.2, only_Q_ratio=0.5), drop_last=True)
    model_on=SBert()
    model_off=SBert()
    model_on.to(device)
    model_off.to(device)

    # model_on.load_state_dict(torch.load('save/save_080.pt', 'cpu'))

    momentum_update(model_on, model_off, 1) #full copy online to offline

    bank=memory(8192)
    optimizer = torch.optim.AdamW(model_on.parameters(), lr=1e-5, weight_decay=1e-2)
    num_epochs=50
    lr_scher=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, cooldown=10, min_lr=1e-6)

    for i in range(num_epochs):
        trainer(i, model_on, model_off)
        if (i+1)%10==0:
            torch.save(model_on.state_dict(),f'save/save_{i+1:03d}.pt')
    torch.save(model_on.state_dict(),'save/save.pt')
