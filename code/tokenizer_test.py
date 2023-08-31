from transformers import AutoTokenizer


model_dir="weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)


# id=tokenizer('你好，這是一串隨機沒有意義的字串')
# print(id.input_ids)
# for i in range(len(id.input_ids)):
#     output=tokenizer.decode(id.input_ids[i])
#     print(id.input_ids[i], output)

# output=tokenizer.decode(id.input_ids[5:8])
# print(id.input_ids[5:8], output)

# print(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))[3:259]))


# for i in range(256):
#     print(tokenizer.decode(i), end=' ')

id=tokenizer('軟題軟體')
print(id.input_ids)
