# Reproduce-DPO
This repo is meant to be a from-scratch implementation to reproduce the IMDB Experiment in the DPO paper

1. The first step is to run `generate_data.py`. According to the paper, they first SFT a base gpt2-large on IMDB for one epoch, and they also provide the model checkpoint. Instead, I decided to use a well-known gpt-imdb model from lvwerra. As far as I can tell, there are no discernible differences to the performance. Otherwise, the data generation process is exactly as the paper describes, where we end up with 4 samples for each of the 25k prefixes.

2. The second step is to run `sample_preference_pairs.py`. There is no difference between the code here and the paper, and we end up with 6 preference pairs for each of the 25k prefixes/responses we generated, totalling to 150k rows of data.

3. The third step is to run DPO. This can be done on your favorite training library. Here, I do it on TRL, and write a short eval function to plot the rewards against KL averaged over the entire eval dataset. For reference, I split my data into 100k training data and 50k for evaluation.
