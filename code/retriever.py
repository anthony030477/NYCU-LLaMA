import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import trainDataset
from model import SBert
from utils import cos_sim


@torch.inference_mode()
def get_feature(model, texts)->torch.Tensor:
    '''
    texts: text list with shape:(N)\\
    return: tensor with shape:(N, 768)
    '''
    feature_list=[]
    text_list=[]
    model.eval()

    dataloader = DataLoader(texts, batch_size=32, shuffle=False)
    for texts in (bar:=tqdm(dataloader,ncols=0)):
        bs=len(texts)

        feature  = model(texts)#(bs, d)
        feature_list.append(feature)
        text_list.extend(texts)


    feature_list=torch.cat(feature_list)



    #feature and text


    data_dict = {
    'feature': feature_list,
    'text': text_list
    }
    return  data_dict


class Retriever(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model=SBert()
        self.model.eval()

    def build_index(self, texts:list[tuple[str]]):
        texts=list(filter(lambda x:type(x[0])==str and type(x[1])==str, texts))
        self.Q = [x[0] for x in texts]
        self.A = [x[1] for x in texts]
        dict = get_feature(self.model, self.Q)
        self.feature=dict['feature']

    @torch.no_grad()
    def retrieve(self, query:str, k=5, threshold=0.8):
        '''
        return k retrieved id and similarity
        '''
        query_feature = self.model(query)
        if len(query_feature.shape)==1:
            query_feature=query_feature[None,:]
        #cosine similarity
        sim = cos_sim(query_feature, self.feature)[0]

        #top-k vector and index
        v, id = torch.topk(sim, k, dim=0, largest=True)
        return [(self.Q[idx], self.A[idx], int(sim*100)) for idx, sim in zip(id[v>threshold], v[v>threshold])]







