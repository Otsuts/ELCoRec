from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import sys
import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default="GAARA_RRAP_15")
parser.add_argument("--log", type=str, default="logs")
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--model_path", type=str, default="./models/vicuna")
parser.add_argument("--model", type=str, default="vicuna")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=15942)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1)

# Here are args of prompt
parser.add_argument("--dataset", type=str, default="BookCrossing")
parser.add_argument("--K", type=int, default=60)


args = parser.parse_args()
args.output_path = f"trained_models/{args.dataset}/{args.data_type}_GAARA_trained_models/{args.train_size}"

torch.cuda.empty_cache()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

assert args.dataset in ["ml-1m", "AZ-Toys", "ml-25m", "BookCrossing"]

data_path = f"./data/{args.dataset}/proc_data/data"

args.per_device_eval_batch_size = 1


print("*" * 100)
print(args)
print("*" * 100)

transformers.set_seed(args.seed)

print(f"Shot: {args.train_size}")
print(f"Samples used: {args.train_size}")

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"


MICRO_BATCH_SIZE = args.per_device_eval_batch_size
BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
LEARNING_RATE = args.lr
CUTOFF_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size  # 2000
USE_8bit = True
OUTPUT_DIR = args.output_path

if USE_8bit is True:
    warnings.warn(
        "If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2"
    )

TARGET_MODULES = [
    "q_proj",
    "v_proj",
]


DATA_PATH = {
    "train": "/".join(
        [data_path, f"train/train_{args.data_type}_{args.train_size}.json"]
    )
}


device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(args.model_path)

# Model
from mymodel import LLM4Rec

model = LLM4Rec(
    input_dim=32,
    output_dim=5120,
    load_in_8bit=USE_8bit,
    use_lora=args.use_lora,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=TARGET_MODULES,
    model_path=args.model_path,
    train_size=args.train_size,
    K=args.K,
    set="train",
    dataset=args.dataset,
)

data = load_dataset("json", data_files=DATA_PATH)
# data["train"] = data["train"].select(range(args.train_size))
print("Data loaded.")


now_max_steps = max((len(data["train"])) // BATCH_SIZE * EPOCHS, EPOCHS)
MAX_STEPS = now_max_steps


def generate_and_tokenize_prompt(
    data_point,
    index,
):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {data_point['input']} ASSISTANT: "
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    """
        )
    )
    unk_ = model.llama_tokenizer.unk_token
    user_prompt = user_prompt.replace("<UserID>", unk_)
    len_user_prompt_tokens = (
        len(
            model.llama_tokenizer(
                user_prompt,
                truncation=True,
                max_length=2048 + 1,
            )["input_ids"]
        )
        - 1
    ) - 1  # no eos token
    full_tokens = model.llama_tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=2048 + 1,
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
        "gat_input": (torch.tensor(index)),
    }


train_data = data["train"].map(generate_and_tokenize_prompt, with_indices=True)
train_data = train_data.remove_columns("output")
train_data = train_data.remove_columns("input")
print("Data processed.")


def compute_metrics(eval_preds):
    try:
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        ll = log_loss(pre[1], pre[0])
        acc = accuracy_score(pre[1], pre[0] > 0.5)
        return {
            "auc": auc,
            "ll": ll,
            "acc": acc,
        }
    except:
        return {
            "auc": 0.114514,
            "ll": 0.114514,
            "acc": 0.114514,
        }


def preprocess_logits_for_metrics(logits, labels):
    """
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 3869, labels == 1939))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)
    labels_index[:, 1] = labels_index[:, 1] - 1

    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold


trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="epoch",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=30,
        load_best_model_at_end=False,
        metric_for_best_model="eval_auc",
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
        remove_unused_columns=False,  # Set to False when debugging
        label_names=["labels"],
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        model.llama_tokenizer, return_tensors="pt", padding="longest"
    ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
model.llama_model.config.use_cache = False

# if args.use_lora:
#     old_state_dict = model.llama_model.state_dict
#     model.llama_model.state_dict = (
#         lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
#     ).__get__(model.llama_model, type(model.llama_model))

# if torch.__version__ >= "2" and sys.platform != "win32":
#     model.llama_model = torch.compile(model.llama_model)


print("Start training...")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.llama_model.save_pretrained(OUTPUT_DIR)
# model_path = os.path.join(OUTPUT_DIR, "adapter.pth")
# embedding_proj = model.embedding_proj.state_dict()
# torch.save({"embedding_proj": embedding_proj}, model_path)
