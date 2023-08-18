import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from transformers import LlamaTokenizer, LlamaForCausalLM


class Roberta(torch.nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        self.model =  RobertaModel.from_pretrained("roberta-base")
        self.outlayer=nn.Sequential(#nn.Dropout(p=0.1),
                                    # nn.Linear(768, 1024),
                                    # nn.LeakyReLU(),
                                    # # nn.Dropout(p=0.1),
                                    # nn.Linear(1024, 2048),
        )
        self.expander=nn.Sequential(#nn.Dropout(p=0.1),
                                    nn.Linear(768, 1024),
                                    # nn.LeakyReLU(),
                                    # #nn.Dropout(p=0.1),
                                    # nn.Linear(2048, 5096),
                                    # #nn.Dropout(p=0.1),
                                    # nn.LeakyReLU(),
                                    # nn.Linear(5096, 8192),
        )
        self.mlp=nn.Linear(768,3)
    def forward(self, input_ids,attention_mask):
        x=self.model(input_ids=input_ids,attention_mask=attention_mask)
        # x=self.outlayer(x.last_hidden_state[:,0,:])
        feature=x.last_hidden_state[:,0,:]
        x=self.expander(x.last_hidden_state[:,0,:])
        # predict=self.mlp(feature)
        return feature,x#,predict

class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, size = 100, model_dim = 3200):
        super().__init__()
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

    def forward(self, x):

        ff_out = self.adapter_block(x)
        # Skip connection
        adapter_out = ff_out + x

        return adapter_out


class Adaptered(nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter()

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        output = (self.adapter(orig_out[0].unsqueeze(0))[0],)

        return output
    
class LLaMA(torch.nn.Module):
    def __init__(self):
        super(LLaMA, self).__init__()
        model_path = 'openlm-research/open_llama_3b_v2'
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,)   
        
        for params in self.model.parameters():
            params.requires_grad = False
        for i in range(26):
            self.model.model.layers[i].self_attn.o_proj = Adaptered(self.model.model.layers[i].self_attn.o_proj)
            # self.model.model.layers[i].mlp.gate_proj = Adaptered(self.model.model.layers[i].mlp.gate_proj)
    def forward(self, inputs,masks,labels):
        x=self.model(input_ids=inputs,attention_mask=masks,labels=labels)
        # x=self.model.model.embed_tokens(inputs)
        
        # for i in range(26):
        #     x=self.model.model.layers[i].self_attn(input_ids=inputs,attention_mask=masks,labels=labels)
        #     x=self.adapter(x)
        #     x=self.model.model.layers[i].mlp(x)
        #     x=self.model.model.layers[i].input_layernorm(x)
        #     x=self.model.model.layers[i].post_attention_layernorm(x)
        # x=self.model.model.norm(x)
        # x=self.model.model.lm_head(x)
        return x.logits,x.loss

if __name__=='__main__':
    # tokenizer =RobertaTokenizer.from_pretrained("roberta-base")
    # text='I love deep learning!'
    # encoded_input = tokenizer(text, return_tensors='pt')
    # # print(encoded_input)
    
    # model=Roberta()
    # print(model)
    # exit()
    # feature,x,_=model(**encoded_input)
    # print(x.shape)
    tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_3b_v2')
    text='I love deep learning!'
    inputs=tokenizer(text,return_tensors="pt")
    # print(inputs['input_ids'],inputs['attention_mask'])
    model=LLaMA()
    model.to('cuda')
    # print(model)
    outputs=model(inputs['input_ids'].cuda(),inputs['attention_mask'].cuda(),inputs['input_ids'].cuda())
    
    print(outputs[1].shape)