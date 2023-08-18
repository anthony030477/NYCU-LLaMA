import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import LlamaTokenizer, LlamaForCausalLM

class trainDataset(Dataset):
    def __init__(self,sub_glue='wnli'):
        self.dataset = load_dataset("glue", sub_glue)
    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        premise=self.dataset['train'][idx]['sentence1']
        hypothesis=self.dataset['train'][idx]['sentence2']
        label=self.dataset['train'][idx]['label']
        return premise,hypothesis ,label
    
class trainDataset_QA(Dataset):
    def __init__(self,):
        self.dataset = load_dataset("trivia_qa",'rc')
    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        question=self.dataset['train'][idx]['question']
        # print(self.dataset['train'][idx].keys())
        answer=self.dataset['train'][idx]['answer']['value']
        return question,answer
    
class trainDataset_NEWS(Dataset):
    def __init__(self,):
        self.dataset= load_dataset("zeroshot/twitter-financial-news-sentiment")
    def __len__(self):
        return len(self.dataset['train'])
    def __getitem__(self, idx):
        text=self.dataset['train'][idx]['text']
        label=self.dataset['train'][idx]['label']
        return text,label


class testDataset_NEWS(Dataset):
    def __init__(self,):
        self.dataset= load_dataset("zeroshot/twitter-financial-news-sentiment")
    def __len__(self):
        return len(self.dataset['validation'])
    def __getitem__(self, idx):
        text=self.dataset['validation'][idx]['text']
        label=self.dataset['validation'][idx]['label']
        return text,label


def collect_fn(batch):
    '''
    return:torch.tensor(token_id) batchxseq_len 
    '''
    # bodys,titles=zip(*batch)
    input_list=[]
    
    label_list=[]
    mask_list=[]
    tokenizer =RobertaTokenizer.from_pretrained("roberta-base")
    for premise,hypothesis ,label in batch:
        text='Premise: '+premise+'Hypothesis: '+hypothesis
        input_list.append(text)
        label_list.append(label)
   
    output=tokenizer(text=input_list,return_tensors="pt",padding=True,truncation=True,max_length=512)
    
    return output.input_ids , output.attention_mask , label_list
def collect_fn_news(batch):
    '''
    return:torch.tensor(token_id) batchxseq_len 
    '''
    # bodys,titles=zip(*batch)
    input_list=[]
    
    label_list=[]
    mask_list=[]
    tokenizer =RobertaTokenizer.from_pretrained("roberta-base")
    for text,label in batch:
        text='News: '+text+'Sentiment: '
        input_list.append(text)
        label_list.append(label)
   
    output=tokenizer(text=input_list,return_tensors="pt",padding=True,truncation=True,max_length=512)
    labels=torch.tensor(label_list)
    return output.input_ids , output.attention_mask , labels,input_list

def collect_fn_llama(batch):
    '''
    batch: question,answer
    return:torch.tensor(token_id) batchxseq_len 
    '''
    # bodys,titles=zip(*batch)
    max_p_len=int(512*0.7)
    max_c_len=512-max_p_len
    tokens=[]
    masks=[]
    targets=[]
    tokenizer =LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
    for p,c in batch:
        p_out=tokenizer(p, return_tensors="pt", truncation=True, max_length=max_p_len-1)
        c_out=tokenizer(c, return_tensors="pt", truncation=True, max_length=max_c_len)
        #print(p_out.input_ids.shape)#(1, len)

        p_ids =torch.cat([p_out['input_ids'][0],torch.tensor([tokenizer.eos_token_id])])
        
        p_mask=torch.ones_like(p_ids)

        c_ids =torch.cat([c_out['input_ids'][0],torch.tensor([tokenizer.eos_token_id])])[1:]
       
        c_mask=torch.ones_like(c_ids)

        ids=torch.cat([p_ids,c_ids])

        tokens.append(ids)
        masks.append(torch.cat([p_mask,c_mask]))
        targets.append(torch.cat([torch.ones_like(p_ids)*-100, c_ids]))

    tokens=pad_sequence(tokens, batch_first=True, padding_value=tokenizer.eos_token_id)
    masks=pad_sequence(masks, batch_first=True)
    targets=pad_sequence(targets, batch_first=True, padding_value=-100)
    return tokens , masks , targets


if __name__=='__main__':
    dataset=trainDataset_QA()
    print(dataset[0])
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True,collate_fn=collect_fn_llama)
    for ids,masks ,labels,_ in train_dataloader:
        print(ids.shape)
        # print(ids)
        print(labels.shape)
        # print(labels)
        print(masks.shape)
        # print(masks)
        exit()