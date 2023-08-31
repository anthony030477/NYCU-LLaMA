
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
from retriever import Retriever

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sep='</s>'
eos='</s>'
class TaiwanLLaMaGPTQ:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        eos_id=self.tokenizer.eos_token_id
        self.eos=self.tokenizer.eos_token
        new_line=[[self.tokenizer('\n\n').input_ids[-1]]]
        unicode=[[i] for i in range(3,259)]
        self.generate_config=dict(
                            max_new_tokens = 1024,
                            # top_k = 50,
                            # top_p = 0.9,
                            # temperature  =  0.7,
                            no_repeat_ngram_size = 20,
                            do_sample  = False,
                            num_beams = 1,
                            bad_words_ids = None
                            )
        self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=True,
            strict=False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=''

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.thread_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
    def get_prompt(self, message: str, chat_history: list[tuple[str, str]]) -> str:
        texts = [f'{self.system_prompt}']
        if self.external is not None:
            texts.append('<fact>'+self.external+'<end of fact>###')
        for user_input, response in chat_history[-3:]:
            texts.append('USER: '+user_input.strip())
            texts.append('ASSISTANT: '+response.strip())
        texts.append('USER: '+message.strip())
        texts.append('ASSISTANT: ')
        return (sep).join(texts)

    def generate(self, message: str):
        prompt = self.get_prompt(message, self.chat_history)
        # print('ture input:\n', prompt)
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

    dataset=trainDataset()
    dataset+=[('陽明交大有校長嗎?','校長是林奇宏')]
    dataset+=[('陽明交大有校狗嗎?','有很多校狗')]
    dataset+=[('陽明交大校長是誰?','校長是林奇宏')]

    R = Retriever()
    R.to(device)
    R.build_index(dataset)
    s = ''
    # test_text=['要怎麼下載學校的軟體?','課外活動輔導組在哪裡?','請問我要怎麼申請就學貸款']


    while True:
        s = input("User: ")

        answer =R.retrieve(query = s, k=5, threshold=0.8)
        # print('-'*50)
        if len(answer)>0:
            inferencer.external=''
        for Q, A, sim in answer:
            # inferencer.system_prompt+=f'Q: {dataset[j[1].item()][0] }\nA: {dataset[j[1].item()][1]}\n'
            inferencer.external+=f'###{Q}\n{A}\n'+eos

        inferencer.system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant base on fact gives helpful, detailed, and polite answers. Without any link or phone number. Use 繁體中文 in detail.\n"#You are built by NTU Miulab by Yen-Ting Lin for research purpose.

        if s != '':
            print('-'*60+'\n'+inferencer.external+'\n'+'-'*60)
            print('Assistant: ')
            inferencer.generate(s)

