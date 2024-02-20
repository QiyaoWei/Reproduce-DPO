# Reproduce-DPO
This repo is meant to be a from-scratch implementation to reproduce the IMDB Experiment in the DPO paper

1. The first step is to run `generate_data.py`. According to the paper, they first SFT a base gpt2-large on IMDB for one epoch, and they also provide the model checkpoint. Instead, I decided to use a well-known gpt-imdb model from lvwerra. As far as I can tell, there are no discernible differences to the performance. Otherwise, the data generation process is exactly as the paper describes.
