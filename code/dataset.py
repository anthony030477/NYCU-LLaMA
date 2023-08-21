import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import LlamaTokenizer, LlamaForCausalLM

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
    return dataset

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
    dataset=trainDataset()
    print(dataset[0])
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True,collate_fn=collect_fn)
    for ids,masks ,labels,_ in train_dataloader:
        print(ids.shape)
        # print(ids)
        print(labels.shape)
        # print(labels)
        print(masks.shape)
        # print(masks)
        exit()