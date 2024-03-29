import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import os
os.environ["WANDB__SERVICE_WAIT"]="6000"

# Step 1:
import pandas as pd
import random
random.seed("1234")
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, pipeline
set_seed(1234)
from datasets import load_dataset
from torch.utils.data import DataLoader
from trl.core import LengthSampler

model_name="lvwerra/gpt2-imdb"
batch_size = 1
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
reward_pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")
reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

def build_dataset(dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataset))

gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 20}

final = list()
# TODO: Check padding, although I don't think gpt2 has padding?
def outputs_and_rewards(query, query_tensor, temperature):
    
    output = model.generate(query_tensor, temperature=temperature, **gen_kwargs).squeeze()[-gen_kwargs["max_new_tokens"]:]
    
    response = tokenizer.decode(output)

    texts = [q + r for q, r in zip(query, response)]
    for output in reward_pipe(texts, **reward_kwargs):
        if output[1]["label"] == "POSITIVE":
            rewards = output[1]["score"]
        else:
            assert output[1]["label"] == "NEGATIVE"
            rewards = output[0]["score"]
            
    return response, rewards

for batch in dataloader:
    df = dict()
    df["query"] = batch["query"][0]
    query_tensor = batch["input_ids"].to("cuda")

    response1, rewards1 = outputs_and_rewards(df["query"], query_tensor, 0.2)
    response2, rewards2 = outputs_and_rewards(df["query"], query_tensor, 0.4)
    response3, rewards3 = outputs_and_rewards(df["query"], query_tensor, 0.6)
    response4, rewards4 = outputs_and_rewards(df["query"], query_tensor, 0.8)
    
    assert type(df["query"]) == type("")
    assert type(response1) == type("")
    assert type(response1) == type("")
    assert type(response2) == type("")
    assert type(response3) == type("")
    assert type(response4) == type("")
    assert type(rewards1) == type(0.0)
    assert type(rewards2) == type(0.0)
    assert type(rewards3) == type(0.0)
    assert type(rewards4) == type(0.0)
    
    
    df["response1"] = response1
    df["reward1"] = rewards1
    df["response2"] = response2
    df["reward2"] = rewards2
    df["response3"] = response3
    df["reward3"] = rewards3
    df["response4"] = response4
    df["reward4"] = rewards4
    final.append(df)

df_results = pd.DataFrame(final)
df_results.to_csv("results.csv", index=False)

# Step 2:
import json
import random
random.seed("1234")
from transformers import set_seed
set_seed(1234)
import pandas as pd
final = list()
data = pd.read_csv("results.csv")

for i in range(len(data)):
    
    d = {0: data["response1"].iloc[i],
         1: data["response2"].iloc[i],
         2: data["response3"].iloc[i],
         3: data["response4"].iloc[i]}

    l = [data["reward1"].iloc[i], data["reward2"].iloc[i], data["reward3"].iloc[i], data["reward4"].iloc[i]]
    indices = sorted(range(len(l)), key=lambda k: l[k])
    
    final.append({"prompt": data["query"][i], "chosen": d[indices[0]], "rejected": d[indices[1]]})
    final.append({"prompt": data["query"][i], "chosen": d[indices[0]], "rejected": d[indices[2]]})
    final.append({"prompt": data["query"][i], "chosen": d[indices[0]], "rejected": d[indices[3]]})
    final.append({"prompt": data["query"][i], "chosen": d[indices[1]], "rejected": d[indices[2]]})
    final.append({"prompt": data["query"][i], "chosen": d[indices[1]], "rejected": d[indices[3]]})
    final.append({"prompt": data["query"][i], "chosen": d[indices[2]], "rejected": d[indices[3]]})
    
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=4)
