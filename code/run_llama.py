
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
import torch

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
                            max_new_tokens = 512,
                            top_k = 50,
                            top_p = 0.9,
                            temperature  =  0.7,
                            no_repeat_ngram_size = 20,
                            do_sample  = False,
                            num_beams = 5,
                            bad_words_ids = None
                            )
        self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=True,
            use_cache=True,
            strict=False)
        self.chat_history = []
        self.system_prompt = ''
        self.external=''
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True)
        self.thread_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def generate(self, message: str):
        prompt = message
        print('ture input:\n', prompt)
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids
        generate_ids = self.model.generate(input_ids=tokens.cuda(), streamer=None, **self.generate_config)
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

    while True:
        s = input("User: ")
        if s != '':
            print(inferencer.generate(s))

