import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn,trainDataset_NEWS,collect_fn_news,testDataset_NEWS
from model import Roberta
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
def featurer(feature_list,text_list):

    '''
    text_input: text list
    return: feature list
    '''
    model.eval()
    for input_ids,input_masks ,labels,text in (bar:=tqdm(train_dataloader,ncols=0)):
        input_ids=input_ids.to(device)
        labels=labels.to(device)
        input_masks=input_masks.to(device)

        feature,_  = model(input_ids,input_masks)#batchsizex768
        for item in range(input_ids.shape[0]):
            feature_list.append(feature[item].cpu().numpy())
            text_list.append(text[item])



if __name__=='__main__':
    dataset=trainDataset_NEWS()

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False,collate_fn=collect_fn_news)
    model=Roberta()
    model.load_state_dict(torch.load('/home/anthony/work/save/save_26.1.pt'))
    model.to(device)
    feature_list=[]
    text_list=[]
    # featurer(feature_list,text_list)
    dataset=testDataset_NEWS()
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=collect_fn_news)

    haha = torch.load('/home/anthony/work/feature.pt')
    ha = torch.stack([torch.from_numpy(val) for val in haha['feature']])

    for input_ids,input_masks ,labels,text in test_dataloader:
        with torch.no_grad():
            fe, _ = model(input_ids.cuda(),input_masks.cuda())
            print(labels)
            print(text)
            break

    hahaha = ha.cuda() @ fe.T

    xxxx = torch.topk(hahaha.squeeze(), 5)

    print(xxxx)

    print('retrive text')

    for i in xxxx.indices:
        print(haha['text'][i])
        print(haha['label'][i])

    '''
    save feature and text
    '''

    # torch.save({'feature': feature_list, 'text': text_list}, 'feature.pt')

    print('finish!')
