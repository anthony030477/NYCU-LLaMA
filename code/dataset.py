from typing import Any
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import pandas as pd



def trainDataset(path='data/'):
    dataframes=[]

    for data in os.listdir(path):
        files_ls=os.listdir(path+'/'+data+'/'+'1120802')
        for files in files_ls:
            if 'QA' in files:
                dataframes.append(pd.read_excel(path+'/'+data+'/'+'1120802'+'/'+files))

    dataset=[]
    for dataframe in dataframes:
        Q=dataframe["問題"].tolist()
        A=dataframe["回答內容"].tolist()
        dataset.extend(zip(Q,A))
    return list(filter(lambda x:type(x[0])==str and type(x[1])==str,dataset))


class collect_fn:
    def __init__(self, max_len=512, drop=0.1, only_Q_ratio=0.3):
        self.max_len=max_len
        self.drop = drop
        self.only_Q_ratio=only_Q_ratio
        self.tokenizer =RobertaTokenizer.from_pretrained("roberta-base")

    def __call__(self, batch):
        '''
        input: question,answer
        return:torch.tensor(token_id) batchxseq_len
        '''
        # bodys,titles=zip(*batch)
        input_list_a=[]
        input_list_b=[]

        onlyQ=torch.rand(1)<self.only_Q_ratio
        for QA in batch:
            text=None
            if type(QA)==tuple or type(QA)==list:
                question, answer=QA
                if type(question)==str and type(answer)==str:
                    text='Question: '+question
                    if not onlyQ:
                        text +='Answer: '+answer
            else:
                text='Question: '+QA

            if text is not None:

                input_list_a.append(text if torch.rand(1)>self.drop else self.text_aug(text))
                input_list_b.append(text if torch.rand(1)>self.drop else self.text_aug(text))



        output_a=self.tokenizer (text=input_list_a,return_tensors="pt",padding=True,truncation=True,max_length=self.max_len)
        output_b=self.tokenizer (text=input_list_b,return_tensors="pt",padding=True,truncation=True,max_length=self.max_len)


        ids_a = output_a.input_ids.clone()
        ids_b = output_b.input_ids.clone()

        if self.drop==0:
            return ids_a, output_a.attention_mask, input_list_a

        return ids_a, ids_b, output_a.attention_mask, output_b.attention_mask,  input_list_a, input_list_b

    def text_aug(self, text):
        #random mask word
        n=len(text)
        rand=torch.randint(10,n,size=[int(n*self.drop)])
        rand=sorted(rand)
        rand=[0]+rand +[n+1]
        texts=''
        for i in range(len(rand)-1):
            texts+=text[rand[i]:rand[i+1]-1]
            if i<len(rand)-2 and torch.rand(1)<0.3:
                texts+=self.tokenizer.mask_token
            if torch.rand(1)<0.2:
                texts+=self.tokenizer.mask_token*2
        return texts

if __name__=='__main__':
    dataset=trainDataset()
    print(dataset[0])
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True,collate_fn=collect_fn())
    for ids,masks in train_dataloader:
        print(ids.shape)
        print(masks.shape)
        exit()
