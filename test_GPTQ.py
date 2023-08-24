
from transformers import AutoTokenizer,TextStreamer,TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM


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
        self.system_prompt = '''Q:  要怎麼登入學校Email信箱收信? A: 學校提供的Email帳號是一個google帳號，直接在gmail登入頁面輸入完整帳號密碼後即可登入使用。
Q:  要怎麼下載/安裝學校的授權軟體? A: 下載前需先安裝下載工具(FileZilla)，詳細說明請參考https://ca.nycu.edu.tw/download/。
各軟體之安裝說明，請參考https://ca.nycu.edu.tw/installation/
Q:  國科會計畫沒通過怎麼辦? A: 可以申請研發處協成型研究計畫，申請資格為本校專任教師及編制內研究人員，近3年內曾發表過論文(含會議論文)、學術專書或專書章節，且本年度申請國科會計畫未通過，且未有其他計畫經費。
Q:  可以自己找實習機構嗎? A: 專業實習機構的選定，須符合學系所訂的篩選條件及經過評估後，送請實習委員會同意，同學可以建議學系評估。
Q:  原陽明入口網忘記密碼怎麼辦? A: 1.點選忘記密碼，以留存E_Mail帳號修改密碼
 2.本人來電至資訊中心分機 123詢問，核對身份將密碼設回預設值(有權限）
  無重設帳號密碼權限人員，請使用者提供(1)學號/人事代號(2)證件(3)連絡電話，寄信到icc@nycu.edu.tw。確認身份後將協助重設。
  。\n\n給定上述資訊，回答使用者的問題。如果上述資料不足以回答，說不知道。'''

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

inferencer = TaiwanLLaMaGPTQ("weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ")


s = ''
while True:
    s = input("User: ")
    if s != '':
        # print ('Answer:')
        # for t in inferencer.thread_generate(s):
        #     print(t, end="")
        print(inferencer.generate(s))
        # print ('-'*80)

