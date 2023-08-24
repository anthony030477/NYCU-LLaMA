
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
from dataset import trainDataset,collect_fn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Roberta, SBert
import numpy as np
from utils import cos_sim
from inference import featurer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TaiwanLLaMaGPTQ:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        eos_id=self.tokenizer.eos_token_id
        new_line=self.tokenizer('\n\n').input_ids[-1]
        self.generate_config=dict(max_new_tokens = 1024,
                              top_k = 50,
                              no_repeat_ngram_size=   8,
                            #   temperature  =  0.5,
                              do_sample  = False,
                              num_beams = 1,
                              eos_token_id= [eos_id],
                              bad_words_ids=[[new_line]]
                            )
        self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=True,
            strict=False)
        self.chat_history = []
        self.system_prompt = ''

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.thread_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
    def get_prompt(self, message: str, chat_history: list[tuple[str, str]]) -> str:
        texts = [f'<<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n']
        for user_input, response in chat_history:
            texts.append(f'user:\n{user_input.strip()}\nsystem:\n{response.strip()}\n')
        texts.append(f'user:\n{message.strip()}\nsystem:\n')
        return ''.join(texts)

    def generate(self, message: str):
        prompt = self.get_prompt(message, self.chat_history)
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids
        generate_ids = self.model.generate(input_ids=tokens.cuda(), streamer=self.streamer, **self.generate_config)
        output = self.tokenizer.decode(generate_ids[0, len(tokens[0]):-1]).strip()
        self.chat_history.append([message, output])
        return output

    def thread_generate(self, message:str):
        from threading import Thread
        prompt = self.get_prompt(message, self.chat_history)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        generation_kwargs = dict(
            inputs=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask,
            temperature=0.1,
            max_new_tokens=1024,
            streamer=self.thread_streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in self.thread_streamer:
            yield new_text

        thread.join()


if __name__=='__main__':
    inferencer = TaiwanLLaMaGPTQ("weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ")

    model=SBert()
    model.to(device)
    model.eval()
    dataset=trainDataset()
    feature = featurer(model, dataset)

    s = ''
    # test_text=['要怎麼下載學校的軟體?','課外活動輔導組在哪裡?','請問我要怎麼申請就學貸款']


    while True:
        s = input("User: ")
        test_dataloader = DataLoader([s], batch_size=100, shuffle=False,collate_fn=collect_fn(drop=0))


        for input_ids,_, input_masks ,text in test_dataloader:
            with torch.no_grad():
                test_feature= model(text)

        sim = cos_sim(test_feature, feature)
        vs, ids = torch.topk(sim, 5, dim=1, largest=True)


        # print('-'*50)
        inferencer.system_prompt=''
        for j in zip(vs[0], ids[0]):
            inferencer.system_prompt+=f'Q: {dataset[j[1].item()][0] } A: {dataset[j[1].item()][1]}\n'
            # print('Q: ',dataset[j[1].item()][0], 'A:', dataset[j[1].item()][1] )#j[0].item()

        inferencer.system_prompt+='''\n\nGiven the above information, answer the user's question. If the above information is not enough to answer, say you don't know. Don't show any link or phone number.'''

        if s != '':
            # print ('Answer:')
            # for t in inferencer.thread_generate(s):
            #     print(t, end="")
            print(inferencer.system_prompt)
            inferencer.generate(s)
            # print ('-'*80)

