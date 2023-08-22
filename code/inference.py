import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn
from model import Roberta
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
def featurer(feature_list,text_list):
    model.eval()
    for ids_a, _, input_masks,text in (bar:=tqdm(train_dataloader,ncols=0)):
        ids_a=ids_a.to(device)
        input_masks=input_masks.to(device)

        feature,_  = model(ids_a,input_masks)#batchsizexdim
        for item in range(input_ids.shape[0]):
            feature_list.append(feature[item].cpu().numpy())
            text_list.append(text[item])
    
    '''
    save feature and text 
    '''

    data_dict = {
    'feature': feature,
    'text': text
    }
    torch.save(data_dict, 'feature.pt')
    print("saved~~")


if __name__=='__main__':
    dataset=trainDataset()
    
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False,collate_fn=collect_fn)
    model=Roberta()
    model.load_state_dict(torch.load('/home/anthony/work/save/save_26.1.pt'))
    model.to(device)
    feature_list=[]
    text_list=[]
    featurer(feature_list,text_list)
    dataset=testDataset()
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=collect_fn)

    haha = torch.load('feature.pt')
    ha = torch.stack([torch.from_numpy(val) for val in haha['feature']])

    for input_ids,input_masks ,text in test_dataloader:
        with torch.no_grad():
            test_feature, _ = model(input_ids.cuda(),input_masks.cuda())
            
            break

    hahaha = ha.cuda() @ test_feature.T

    nearst_feature = torch.topk(hahaha.squeeze(), 5)


    print('retrive text')

    for i in nearst_feature.indices:
        print(haha['text'][i])
        print(haha['label'][i])


