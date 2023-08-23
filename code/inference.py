import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn
from model import Roberta
import numpy as np
from utils import cosine_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def featurer(dataset):

    '''
    text_input: text list
    return: feature list
    '''
    feature_list=[]
    text_list=[]
    model.eval()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,collate_fn=collect_fn(drop=0, only_Q_ratio=1))
    for ids_a, _, input_masks,text in (bar:=tqdm(dataloader,ncols=0)):
        ids_a=ids_a.to(device)
        input_masks=input_masks.to(device)

        feature,_  = model(ids_a,input_masks)#(bs, d)
        feature_list.append(feature)
        text_list.extend(text)

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


if __name__=='__main__':
    dataset=trainDataset()

    model=Roberta()
    model.load_state_dict(torch.load('save/save_010.pt'))
    model.to(device)
    # compute latent and save
    # featurer(dataset)

    dataset=['共學之上課方式為何?']
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=collect_fn(drop=0))

    feature_text = torch.load('save/feature.pt')
    feature = feature_text['feature'].to(device)

    for input_ids,_, input_masks ,text in test_dataloader:
        with torch.no_grad():
            test_feature, _ = model(input_ids.to(device),input_masks.to(device))

            break

    sim = cosine_similarity(test_feature, feature)
    nearst_feature = torch.topk(sim, 5, dim=1)


    print('retrive text')

    for i in nearst_feature.indices:
        for j in i:
            print(feature_text['text'][j])

