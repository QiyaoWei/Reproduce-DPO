# Reproduce-DPO
This repo is meant to be a from-scratch implementation to reproduce the IMDB Experiment in the DPO paper

1. According to the paper, they first SFT a base gpt2-large on IMDB for one epoch, and they also provide the model checkpoint. Instead, I decided to use a well-known gpt-imdb model from lvwerra to avoid SFT. Otherwise, the data generation process is exactly as the paper describes, where we end up with 4 samples for each of the ~25k prefixes.

2. The first step in reproducing is to run `generate_preference_pairs.py`. Again, there is no difference between the code here and the paper, and we end up with 6 preference pairs for each of the ~25k prefixes we generated, totalling to ~150k rows of data. Link to my generated data [here](https://huggingface.co/datasets/QiyaoWei/Reproducing-DPO)

3. Now we can train DPO. This can be done on your favorite training library. Here, I do it on TRL, and write a short eval function to plot the rewards against KL averaged over the entire eval dataset. For reference, I split my data into ~100k training data and ~50k for evaluation.

![kl_vs_rewards](https://github.com/QiyaoWei/Reproduce-DPO/assets/36057290/c20811e0-5a71-4a6a-9b6d-d49cb5cc6d9e)

Remark: Why do the rewards not match the paper? I reckon it has something to do with me choosing a lazy ref model. IMHO, what would be the most correct procedure is to (1) generate the preference dataset (2) run SFT on the positive samples (3) run DPO. This is because SFT essentially pushes the model closer to the data distribution, and then it makes sense to do a KL constraint because we don't want the generation to become far away from the data distribution. Any other method may incur an OOD problem.

