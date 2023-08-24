import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

from transformers import AutoModel ,AutoTokenizer
from transformers import AutoTokenizer



@torch.no_grad()
def momentum_update(src:nn.Module, dst:nn.Module, factor=0.01):
    for s, d in zip(src.parameters(), dst.parameters()):
        d.data = (1-factor)*d.data + factor*s.data

class Roberta(torch.nn.Module):
    def __init__(self, out_dim=2048):
        super(Roberta, self).__init__()
        self.model =  RobertaModel.from_pretrained("roberta-base")
        self.projecter = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 1024, bias=True),
            nn.LeakyReLU(),
            nn.Linear(1024, out_dim, bias=False)
        )
        # self.projecter =nn.Linear(768, out_dim, bias=False)
        self.predicter = nn.Linear(out_dim, out_dim, bias=False)



    def forward(self, input_ids,attention_mask):
        x=self.model(input_ids=input_ids, attention_mask=attention_mask)
        feature=x.last_hidden_state[:,0,:]
        del x

        projection = self.projecter(feature)
        prediction = self.predicter(projection)
        return projection, prediction

class SBert(torch.nn.Module):
    def __init__(self, out_dim=2048):
        super(SBert, self).__init__()
        self.model = AutoModel.from_pretrained("deepset/sentence_bert")
        self.projecter = nn.Linear(768, out_dim, bias=False)
        self.predicter = nn.Linear(out_dim, out_dim, bias=False)



    def forward(self, input_ids,attention_mask):
        x=self.model(input_ids=input_ids, attention_mask=attention_mask)
        feature=x.last_hidden_state[:,0,:]
        del x

        projection = self.projecter(feature)
        prediction = self.predicter(projection)
        return projection, prediction
    
class Contriever(torch.nn.Module):
    def __init__(self, out_dim=2048):
        super(Contriever, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.projecter = nn.Linear(768, out_dim, bias=False)
        self.predicter = nn.Linear(out_dim, out_dim, bias=False)
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")


    def forward(self, input_ids,attention_mask):
        x=self.model(input_ids=input_ids, attention_mask=attention_mask)
        feature=x.last_hidden_state[:,0,:]
        del x

        projection = self.projecter(feature)
        prediction = self.predicter(projection)
        return projection, prediction
if __name__=='__main__':
    m1=SBert()
    