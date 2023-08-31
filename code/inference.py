import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset,collect_fn
from model import Roberta, SBert
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

        feature  = model(text)#(bs, d)
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
    model=SBert()
    model.to(device)
    model.eval()
    # compute latent and save
    dataset+=[('陽明交大有校長嗎?','校長是')]
    dataset+=[('陽明交大有校狗嗎?','校狗')]
    dataset+=[('學校有校長嗎?','校長是')]
    dataset+=[('學校有校狗嗎?','校狗')]

    # for q,a in dataset:
    #     if type(q)==str and '校歌'in q:
    #         print(q,a)
    # exit()
    featurer(model, dataset)

    test_text=[
        # '要怎麼下載學校的授權軟體?',
        # '課外活動輔導組在學校的哪裡?',
        # '怎麼申請就學貸款',
        # '電子報的問題',
        '我們學校有校歌嗎?',
        '我們學校有校狗嗎?',
        '我們學校有校草嗎?',
        '我們學校有校花嗎?',
        '有什麼社團',
        '學校有洗衣機嗎',
        '學校有提供校內工讀嗎？',
        '學校有獎學金嗎 ?',
        'Does the school have scholarships?',
        'Do I need to leave Taiwan after graduated?',
        '電機系的修業規章',

        ]
    test_dataloader = DataLoader(test_text, batch_size=100, shuffle=False,collate_fn=collect_fn(drop=0))

    feature_text = torch.load('save/feature.pt')
    feature = feature_text['feature'].to(device)


    for input_ids, input_masks ,text in test_dataloader:
        with torch.no_grad():
            test_feature= model(text)



    sim = cos_sim(test_feature, feature)
    vs, ids = torch.topk(sim, 5, dim=1, largest=True)
    text=feature_text['text']

    print('retrive text')

    for i in range(len(test_text)):
        v= vs[i]
        index=ids[i]
        print('-'*50,'\nTrue query: ', test_text[i], end='\n\n\n')
        for j in zip(v, index):
            if j[0].item()>0.5:
                print('Q: ',dataset[j[1].item()][0], j[0].item() )#'A:', dataset[j[1].item()][1],


