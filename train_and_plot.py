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

# where custom_eval is defined like
# def custom_eval(
#     self, beta,
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

#     all_logits = list()
#     all_rewards = list()
#     reward_pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")
#     reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
#     gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": self.tokenizer.eos_token_id, "max_new_tokens": 20}
#     for step, batch in enumerate(eval_dataloader):

#         (policy_chosen_logps, policy_rejected_logps, _, _,) = self.concatenated_forward(self.model, batch)
#         (reference_chosen_logps, reference_rejected_logps, _, _,) = self.concatenated_forward(self.ref_model, batch)
#         pi_logratios = policy_chosen_logps - policy_rejected_logps
#         ref_logratios = reference_chosen_logps - reference_rejected_logps
#         logits = pi_logratios - ref_logratios
#         encoding = self.tokenizer(batch["prompt"], return_tensors="pt").to("cuda")

#         output = self.model.generate(encoding["input_ids"], **gen_kwargs).squeeze()[-gen_kwargs["max_new_tokens"]:]
#         # outputs = self.tokenizer.batch_decode(self.model.generate(**encoding, max_new_tokens=512), skip_special_tokens=True)
#         response = self.tokenizer.decode(output)
#         rewards = reward_pipe(response, **reward_kwargs)
#         all_logits.append(logits.detach().cpu().item())
#         all_rewards.append(rewards)
#         del policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, pi_logratios, ref_logratios, encoding, output, response, logits, rewards

#     self.log({"kl": all_logits, "rewards": all_rewards})
#     with open(str(beta) + 'logits.json', 'w', encoding='utf-8') as f:
#         json.dump(all_logits, f, ensure_ascii=False, indent=4)
#     with open(str(beta) + 'rewards.json', 'w', encoding='utf-8') as f:
#         json.dump(all_rewards, f, ensure_ascii=False, indent=4)

# And plotting is defined like
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import scienceplots

# betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# total_kl = list()
# total_rewards = list()

# for beta in betas:
    
#     data = open(str(beta) + "rewards.json")
#     data = json.load(data)
#     rewards = list()
#     for i in data:
#         if i[0]["label"] == "POSITIVE":
#             rewards.append(i[0]["score"])
#         else:
#             assert i[1]["label"] == "POSITIVE"
#             rewards.append(i[1]["score"])
#     total_rewards.append(rewards)
    
#     data = open(str(beta) + "logits.json")
#     data = json.load(data)
#     total_kl.append(data)
    
# total_rewards_std = scipy.stats.sem(total_rewards, axis=1)
# total_rewards = np.mean(total_rewards, axis=1)
# total_kl_std = scipy.stats.sem(total_kl, axis=1)
# total_kl = np.mean(total_kl, axis=1)

# plt.style.use("science")
# plt.figure()
# plt.scatter(total_kl, total_rewards)
# plt.xlabel("KL Divergence")
# plt.ylabel("Reward")
# plt.title("IMDb Sentiment Generation")
# plt.savefig("kl_vs_rewards.png")
