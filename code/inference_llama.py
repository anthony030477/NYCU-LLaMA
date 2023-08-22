import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline

# change this to the path of the Taiwan-LLAMA model
model_path = ''

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

demonstration='問題: 要去哪裡看學校的行事曆，要是有問題要問誰?\n回答內容: 學校的行事曆放在學校網頁首頁，右下方白色區域，行事曆負責單位為教務處處本部，本校行事曆皆通過行政會議及教育部核備。 問題: 什麼時候放暑假或是寒假?\n回答內容: 行事曆自112學年度起有標註暑假、寒假開始的日期。\n'

query = '問題:' + '' + '回答內容:'

prompt = [demonstration + query]

# prompt = ["Hey, are you conscious? Can you talk to me?","please say a story"]
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# generation_output = model.generate(
#     input_ids=input_ids, max_new_tokens=100
# )

# print(tokenizer.decode(generation_output[0],skip_special_tokens=True))
tokenizer.pad_token_id=tokenizer.eos_token_id
pipe = pipeline("text-generation", model=model,tokenizer=tokenizer,
                max_length=512, num_return_sequences=1, top_k=50, top_p=0.95,
                temperature=1., num_beams=1, no_repeat_ngram_size=4,
                batch_size=8, early_stopping=False, max_new_tokens=1 ,return_full_text=False,)
# help(pipe)
print(pipe(prompt, do_sample=False,))








# from transformers import AutoTokenizer
# import transformers
# import torch

# model = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# sequences = pipeline(
#     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
