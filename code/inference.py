import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn
from model import Roberta
import numpy as np
from utils import cos_sim
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def featurer(model, dataset):

    '''
    text_input: text list
    return: feature list
    '''
    feature_list=[]
    text_list=[]
    model.eval()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,collate_fn=collect_fn(drop=0, only_Q_ratio=1))
    for ids_a, input_masks, text in (bar:=tqdm(dataloader,ncols=0)):
        bs=ids_a.shape[0]
        ids_a=ids_a.to(device)
        input_masks=input_masks.to(device)

        feature,_  = model(ids_a,input_masks)#(bs, d)
        feature_list.append(feature)
        text_list.extend(text)

        with torch.no_grad():
            sim=cos_sim(feature,feature)

        all_sim=0
        for i in range(bs):
            self_sim = sim[i, i]
            other = sim[i, 0]
            all_sim += self_sim-other
        all_sim/=bs
        # print(all_sim)

    feature_list=torch.cat(feature_list)


    '''
    save feature and text
    '''

    data_dict = {
    'feature': feature_list,
    'text': text_list
    }
    torch.save(data_dict, 'save/feature.pt')
    print("saved~~")
    return feature_list


if __name__=='__main__':
    dataset=trainDataset()
    model=Roberta()
    model.load_state_dict(torch.load('save/save.pt'))
    model.to(device)
    model.eval()
    # compute latent and save
    featurer(model, dataset)

    test_text=['要怎麼下載學校的軟體?','課外活動輔導組在哪裡?','請問我要怎麼申請就學貸款','電子報相關問題']
    test_dataloader = DataLoader(test_text, batch_size=100, shuffle=False,collate_fn=collect_fn(drop=0))

    feature_text = torch.load('save/feature.pt')
    feature = feature_text['feature'].to(device)


    for input_ids, input_masks ,text in test_dataloader:
        with torch.no_grad():
            test_feature, _ = model(input_ids.to(device),input_masks.to(device))



    sim = cos_sim(test_feature, feature)
    vs, ids = torch.topk(sim, 5, dim=1, largest=True)
    text=feature_text['text']
    for i,f in zip(text,sim[3]):
        if type(i)==str and '電子報' in i:
            print(i, f)
            pass

    print('retrive text')

    for v, index in  zip(vs, ids):
        print('-'*50)
        for j in zip(v, index):
            print('Q: ',dataset[j[1].item()][0], 'A:', dataset[j[1].item()][1],j[0].item() )#


