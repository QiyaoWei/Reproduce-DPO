import warnings
from typing import Dict
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import DPOTrainer
from dataclasses import dataclass, field
@dataclass
class ScriptArguments:
    beta: float = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    # args, training_args, model_config = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model_name_or_path = "test" + str(args.beta)
    per_device_train_batch_size = 1
    max_steps = 10000
    gradient_accumulation_steps = 2
    gradient_checkpointing = False
    learning_rate = 1e-4
    report_to = None
    max_length = 1024
    max_prompt_length = 128
    max_target_length = 128
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        max_steps=max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        eval_steps=500,
        output_dir="./test" + str(args.beta),
        optim="rmsprop",
        warmup_steps=150,
        report_to=report_to,
        fp16=True,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset("json", data_files="train.json", split="train")
    eval_dataset = load_dataset("json", data_files="test.json", split="train")
    def split_prompt_ands(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
    # train_dataset = train_dataset.map(split_prompt_ands)
    # eval_dataset = eval_dataset.map(split_prompt_ands)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_target_length=max_target_length,
        max_prompt_length=max_prompt_length,
        generate_during_eval=False,
    )
    wandb.init(project="reproduce-dpo", name="modelbeta=" + str(args.beta))
    trainer.train()
    trainer.save_model("./test" + str(args.beta))
    output_dir = os.path.join("./test" + str(args.beta), "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.custom_eval(str(args.beta))

# # where custom_eval is defined like
# def custom_eval(
#     self, beta, secret_sauce,
#     eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
#     metric_key_prefix: str = "eval",
# ) -> Dict[str, float]:
#     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
#     print(isinstance(eval_dataset, dict))
#     if isinstance(eval_dataset, dict):
#         metrics = {}
#         for eval_dataset_name, _eval_dataset in eval_dataset.items():
#             dataset_metrics = self.custom_eval(
#                 eval_dataset=_eval_dataset,
#                 metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
#             )
#             metrics.update(dataset_metrics)
#         return metrics

#     # memory metrics - must set up as early as possible
#     self._memory_tracker.start()

#     eval_dataloader = self.get_eval_dataloader(eval_dataset)
#     start_time = time.time()

#     all_results = list()
#     reward_pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")
#     reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
#     gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
#                   "pad_token_id": self.tokenizer.eos_token_id, "max_new_tokens": 20, "output_logits": True}
#     for step, batch in enumerate(eval_dataloader):

#         # encoding \in R^{batch_size x input_length}
#         encoding = self.tokenizer(batch["prompt"], return_tensors="pt").to("cuda")
        
#         # generation \in R^{batch_size x output_length}
#         generation_output = self.model.generate(**encoding, pad_token_id=self.tokenizer.eos_token_id)#, return_dict_in_generate=True) #, output_logits=True)
#         abridged_output = generation_output[-gen_kwargs["max_new_tokens"]:]
        
#         # response \in R^{batch_size} (natural language space)
#         response = self.tokenizer.batch_decode(abridged_output, skip_special_tokens=True)
        


#         model_forward = self.model(input_ids=abridged_output)
#         ref_model_forward = self.ref_model(input_ids=abridged_output)
#         # print(model_forward.logits.shape)
#         # print(ref_model_forward.logits.shape)
        
#         model_logps = torch.gather(model_forward.logits[:, :-1, :].log_softmax(-1), dim=2, index=abridged_output[:, 1:].unsqueeze(2)).squeeze(2)
#         ref_model_logps = torch.gather(ref_model_forward.logits[:, :-1, :].log_softmax(-1), dim=2, index=abridged_output[:, 1:].unsqueeze(2)).squeeze(2)
        
#         # calculate kl and reward
#         naive_kl = (model_logps-ref_model_logps).sum().detach().cpu().item()
#         full_kl = torch.nn.functional.kl_div(ref_model_logps, model_logps, log_target=True, reduction="none").sum().detach().cpu().item()
#         # Note: if we want full kl then we need
#         # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
#         # return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

#         rewards = reward_pipe(response, **reward_kwargs)
#         all_results.append({"kl": [naive_kl, full_kl], "rewards": rewards, "response": response})


#     self.log({"results": all_results})
#     with open(str(beta) + str(secret_sauce) + 'results.json', 'w', encoding='utf-8') as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=4)

# # And plotting is defined like
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import scienceplots

# betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ps = [0.0] #[0.0, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001] #"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
# folder_name = "./"

# total_kl = list()
# total_rewards = list()

# for beta in betas:
#     for p in ps:
        
#         data = open(folder_name + str(beta) + str(p) + "results.json")
#         data = json.load(data)
#         rewards = list()
#         kl = list()
#         for i in data:
#             if i["rewards"][0][0]["label"] == "POSITIVE":
#                 rewards.append(i["rewards"][0][0]["score"])
#             else:
#                 assert i["rewards"][0][1]["label"] == "POSITIVE"
#                 rewards.append(i["rewards"][0][1]["score"])
#             kl.append(i["kl"][1])
#         total_rewards.append(rewards)
#         total_kl.append(kl)
#         print(beta, np.mean(rewards), np.mean(total_kl))
    
# total_rewards_std = scipy.stats.sem(total_rewards, axis=1)
# total_rewards = np.mean(total_rewards, axis=1)
# total_kl_std = scipy.stats.sem(total_kl, axis=1)
# total_kl = np.mean(total_kl, axis=1)
# print(total_rewards)
# print(total_kl)

# plt.style.use("science")
# plt.figure()
# plt.scatter(total_kl, total_rewards)
# plt.xlabel("KL Divergence")
# plt.ylabel("Reward")
# plt.title("IMDb Sentiment Generation")
# plt.savefig("kl_vs_rewards.png")
