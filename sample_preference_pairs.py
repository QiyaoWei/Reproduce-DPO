import json
import random
random.seed("1234")
from transformers import AutoTokenizer, set_seed, pipeline
set_seed(1234)
model_name="lvwerra/gpt2-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
reward_pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")
reward_kwargs = {"top_k": None, "batch_size": 256, "truncation": True, "max_length": 512}
final = list()
# Here I only use the 4 out of the 10 generations. Again, I don't believe that there will be a big difference depending on which samples you use.
with open("seven.txt", 'r') as f:
    text = f.read()
    result = text.split("*" * 100)[:-1]
    print(len(result))
    prompt1 = [i.split("*"*99)[0] for i in result]
    response1 = [i.split("*"*99)[1] for i in result]
    reward1 = reward_pipe(response1, **reward_kwargs)
with open("eight.txt", 'r') as f:
    text = f.read()
    result = text.split("*" * 100)[:-1]
    print(len(result))
    prompt2 = [i.split("*"*99)[0] for i in result]
    response2 = [i.split("*"*99)[1] for i in result]
    reward2 = reward_pipe(response2, **reward_kwargs)
with open("nine.txt", 'r') as f:
    text = f.read()
    result = text.split("*" * 100)[:-1]
    print(len(result))
    prompt3 = [i.split("*"*99)[0] for i in result]
    response3 = [i.split("*"*99)[1] for i in result]
    reward3 = reward_pipe(response3, **reward_kwargs)
with open("ten.txt", 'r') as f:
    text = f.read()
    result = text.split("*" * 100)[:-1]
    print(len(result))
    prompt4 = [i.split("*"*99)[0] for i in result]
    response4 = [i.split("*"*99)[1] for i in result]
    reward4 = reward_pipe(response4, **reward_kwargs)

# with open('test.txt', 'w') as f:
#     f.write(str(reward1) + "\n" + str(reward2) + "\n" + str(reward3) + "\n" + str(reward4))
# with open('test.txt', 'r') as f:
#     text = f.read()
#     result = text.split("\n")
#     print(len(result))
# reward1 = eval(result[0])
# reward2 = eval(result[1])
# reward3 = eval(result[2])
# reward4 = eval(result[3])

assert prompt1 == prompt2
assert prompt1 == prompt3
assert prompt1 == prompt4

# We only take the positive score as the reward signal, and we neglect the negative score
reward1 = [i[1]["score"] for i in reward1]
reward2 = [i[1]["score"] for i in reward2]
reward3 = [i[1]["score"] for i in reward3]
reward4 = [i[1]["score"] for i in reward4]

for i in range(len(prompt1)):
    d = {0: response1[i], 1: response2[i], 2: response3[i], 3: response4[i]}
    indices = sorted(range(len([reward1[i], reward2[i], reward3[i], reward4[i]])), key=lambda k: [reward1[i], reward2[i], reward3[i], reward4[i]][k])
    final.append({"prompt": prompt1[i], "chosen_response": d[indices[0]], "rejected_response": d[indices[1]]})
    final.append({"prompt": prompt1[i], "chosen_response": d[indices[0]], "rejected_response": d[indices[2]]})
    final.append({"prompt": prompt1[i], "chosen_response": d[indices[0]], "rejected_response": d[indices[3]]})
    final.append({"prompt": prompt1[i], "chosen_response": d[indices[1]], "rejected_response": d[indices[2]]})
    final.append({"prompt": prompt1[i], "chosen_response": d[indices[1]], "rejected_response": d[indices[3]]})
    final.append({"prompt": prompt1[i], "chosen_response": d[indices[2]], "rejected_response": d[indices[3]]})
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=4)
