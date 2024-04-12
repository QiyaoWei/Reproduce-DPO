import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from tqdm import tqdm
import os
import argparse
import tree


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]



# create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')

args = parser.parse_args()
parent_dir = os.path.dirname(args.checkpoint)
root_dit = ''
step_idxs = args.checkpoint.split('/')[-2]
print(f'Checkpoint: {args.checkpoint}')

print('Saving to:')
print(os.path.join(root_dit, f'{step_idxs}.txt'))
print('*' * 80  + '\n')
path = os.path.join(root_dit, f'{step_idxs}.txt')
if not os.path.exists(path):
    print(f"The file does not exist. Continue running your process.")
    # Insert the code to run your process here
else:
    print(f"The file {path} exists. Exit the process.")
    exit() # Use sys.exit() if this doesn't work


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load your trained model
state_dict_path = args.checkpoint


tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"
if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2-large')
model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))['state'])
# import ipdb; ipdb.set_trace()
model.to('cuda')

# Load reference model
ref_model_name = ''  # this can be changed to another model if needed
ref_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
# ref_tokenizer.truncation_side = "right"
ref_tokenizer.padding_side="left"
if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token=tokenizer.eos_token
ref_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
# ref_model.load_state_dict(torch.load(ref_model_name, map_location=torch.device('cpu'))['state'])
ref_model.to('cuda')
# import ipdb; ipdb.set_trace()

sentiment_fn = pipeline(
    "sentiment-analysis",
    "siebert/sentiment-roberta-large-english",
    top_k=2,
    truncation=True,
    batch_size=64,
    device=model.device  # specify the device id here
)
# Load the imdb dataset
imdb_test = load_dataset("imdb", split="test")

# Preprocess the dataset
eval_prompts = [" ".join(review.split()[:4]) for review in imdb_test["text"]]
inputs = tokenizer(eval_prompts, return_tensors='pt', truncation=True, padding=True)

# Prepare for batching
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], )
print(len(dataset))
data_loader = DataLoader(dataset, batch_size=64)


total_num_items = 0
total_reward = 0

with torch.no_grad():
    for batch_input_ids, batch_attention_mask in tqdm(data_loader):
        # Generate samples from the pretrained model
        # import ipdb; ipdb.set_trace()
        batch_input_ids = batch_input_ids.cuda()
        batch_attention_mask = batch_attention_mask.cuda()
        # with torch.no_grad():
        generated_ids = model.generate(batch_input_ids, attention_mask=batch_attention_mask, do_sample=True, max_new_tokens=60, pad_token_id=tokenizer.pad_token_id)
        
        # Get log probabilities for the generated samples
        
        # with torch.no_grad():
        if True:
            model_inputs = tokenizer(tokenizer.batch_decode(generated_ids), return_tensors='pt', padding=True)
            model_inputs = tree.map_structure(lambda x: x.to(model.device), model_inputs)
            model_outputs = model(**model_inputs, labels=model_inputs['input_ids'])
            model_log_probs = model_outputs.logits.softmax(dim=-1).log()

            ref_inputs = ref_tokenizer(tokenizer.batch_decode(generated_ids), return_tensors='pt', padding=True)
            ref_inputs = tree.map_structure(lambda x: x.to(ref_model.device), ref_inputs)
            ref_outputs = ref_model(**ref_inputs, labels=ref_inputs['input_ids'])
            ref_log_probs = ref_outputs.logits.softmax(dim=-1).log()
        
        generated_ids = model_inputs['input_ids']
        attention_mask = (generated_ids != tokenizer.eos_token_id).float()

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        sentiments = sentiment_fn(generated_texts)
        sentiment_scores = [get_positive_score(sentiment) for sentiment in sentiments]
        # sentiment_scores = [sentiment_fn(text)[0][0]['score'] for text in generated_texts]
        total_reward += sum(sentiment_scores)
        

        total_num_items += len(batch_input_ids)

# Compute averages
average_reward = total_reward / total_num_items
print(f'Averaged reward: {average_reward}')

with open(os.path.join(root_dit, f'{step_idxs}.txt'), 'w') as f: 
    f.write(f'Averaged reward: {average_reward}\n')


