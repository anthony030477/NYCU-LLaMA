import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline
model_path = 'openlm-research/open_llama_3b_v2'
# model_path = 'openlm-research/open_llama_7b'#VRAM不夠

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)
demonstration='Review: Delicious food! Sentiment: Postive\nReview: The food is awful. Sentiment: Negative\nReview: I ordered a steak, but it was overcooked and had a terrible texture. Sentiment: Negative\n'
prompt = [demonstration+'Review: The food at this restaurant is absolutely delicious! Every bite is a burst of flavor. Sentiment:',demonstration+'Review: This smoothie is a taste of summer heaven! The fresh mango flavor is rich and combined with the icy texture, it is incredibly refreshing and delightful. Sentiment:']
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
