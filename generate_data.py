# import random
# random.seed("1234")
# from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
# set_seed(1234)
# from datasets import load_dataset
# from torch.utils.data import DataLoader

# model_name="lvwerra/gpt2-imdb"
# batch_size = 32
# model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# dataset = load_dataset("imdb", split="train")
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(len(dataset))

# # TODO: Check padding
# for batch in dataloader:
#     num_prefix = random.randint(2, 8)
#     encoding = tokenizer(batch["text"], return_tensors="pt", padding=True).to("cuda")
#     encoding = {k: v[:, :num_prefix] for k, v in encoding.items()}
#     prefix = tokenizer.batch_decode(encoding["input_ids"], skip_special_tokens=True)
#     outputs1 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512), skip_special_tokens=True)
#     outputs2 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, num_beams=5), skip_special_tokens=True)
#     outputs3 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, num_beams=5, early_stopping=True), skip_special_tokens=True)
#     outputs4 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, num_beams=5, early_stopping=True, no_repeat_ngram_size=2), skip_special_tokens=True)
#     outputs5 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, do_sample=True, top_k=50), skip_special_tokens=True)
#     outputs6 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95), skip_special_tokens=True)
#     outputs7 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, do_sample=True, top_k=0, temperature=0.2), skip_special_tokens=True)
#     outputs8 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, do_sample=True, top_k=0, temperature=0.4), skip_special_tokens=True)
#     outputs9 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, do_sample=True, top_k=0, temperature=0.6), skip_special_tokens=True)
#     outputs10 = tokenizer.batch_decode(model.generate(**encoding, max_new_tokens=512, do_sample=True, top_k=0, temperature=0.8), skip_special_tokens=True)

#     with open("one.txt", "a") as f:
#         for i in range(len(outputs1)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs1[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("two.txt", "a") as f:
#         for i in range(len(outputs2)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs2[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("three.txt", "a") as f:
#         for i in range(len(outputs3)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs3[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("four.txt", "a") as f:
#         for i in range(len(outputs4)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs4[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("five.txt", "a") as f:
#         for i in range(len(outputs5)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs5[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("six.txt", "a") as f:
#         for i in range(len(outputs6)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs6[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("seven.txt", "a") as f:
#         for i in range(len(outputs7)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs7[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("eight.txt", "a") as f:
#         for i in range(len(outputs8)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs8[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("nine.txt", "a") as f:
#         for i in range(len(outputs9)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs9[i] + "\n")
#             f.write("*" * 100 + "\n")
#     with open("ten.txt", "a") as f:
#         for i in range(len(outputs10)):
#             f.write(prefix[i] + "\n")
#             f.write("*" * 99 + "\n")
#             f.write(outputs10[i] + "\n")
#             f.write("*" * 100 + "\n")
