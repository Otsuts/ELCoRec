import sys

sys.path.append("../")
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
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from utils.support import write_log


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 3869, labels == 1939))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)
    labels_index[:, 1] = labels_index[:, 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold


def get_args():
    parser = argparse.ArgumentParser()
    # Split chunks accroding to the splited json file
    parser.add_argument("--chunk_interval", type=str, default="0:-1")
    parser.add_argument("--data_type", type=str, default="GAARA_RRAP_15")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--log", type=str, default="logs/GAARA")
    parser.add_argument("--output_path", type=str, default="lora-Vicuna")
    parser.add_argument("--model_path", type=str, default="./models/vicuna")
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default="/home/jzchen/ML/newbishe/trained_models/BookCrossing/GAARA_RRAP_15_GAARA_trained_models/15942/checkpoint-62",
    )
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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="BookCrossing")
    parser.add_argument("--test_mode", action="store_true")

    # Args of prompt
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--temp_type", type=str, default="simple")
    return parser.parse_args()


def main(args):
    assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
    ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from peft import (  # Parameter-Efficient Fine-Tuning (PEFT)
        prepare_model_for_int8_training,
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    assert args.dataset in ["ml-1m", "ml-25m", "AZ-Toys", "BookCrossing"]
    data_path = f"./data/{args.dataset}/proc_data/data"
    print("\n")
    print("*" * 50)
    write_log(args, args)
    print("*" * 50)
    print("\n")
    transformers.set_seed(args.seed)
    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
    BATCH_SIZE = min(args.total_batch_size, args.train_size)
    MAX_STEPS = None
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 5  # we don't always need 3 tbh
    LEARNING_RATE = args.lr
    CUTOFF_LEN = 2048  # 256 accounts for about 96% of the data
    LORA_R = 8  # Lora attention dimension
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_SET_SIZE = args.val_size  # 2000
    USE_8bit = True
    if USE_8bit is True:
        warnings.warn(
            "If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2"
        )

    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    DATA_PATH = {
        "test": "/".join(
            [data_path, f"test/test_{args.data_type}_{args.chunk_interval}.json"]
        )
    }
    OUTPUT_DIR = args.output_path  # "lora-Vicuna"
    device_map = "auto"
    # 在单机多卡的情况下，WORLD_SIZE代表着使用进程数量(一个进程对应一块GPU)，这里RANK和LOCAL_RANK这里的数值是一样的，代表着WORLD_SIZE中的第几个进程（GPU
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
    write_log(args.model_path, args)
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
        set="test",
        chunk_interval=args.chunk_interval,
        dataset=args.dataset,
    )

    print("Load lora weights")
    adapters_weights = torch.load(
        os.path.join(args.trained_model_path, "pytorch_model.bin"),
    )
    # **** Important!! Must load in this way!!!****
    state_dict = {
        k: v.cuda()
        for k, v in adapters_weights.items()
        if "lora" in k or "score" in k or "embedding_proj" in k
    }
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded")
    model.eval()

    # 至此，封装后的模型可以作为参数传入 Trainer 类中进行常规训练
    # tokenizer.padding_side = "left"  # Allow batched inference
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

    # Load data
    data = load_dataset("json", data_files=DATA_PATH)
    write_log("Data loaded.", args)
    test_data = data["test"].map(generate_and_tokenize_prompt, with_indices=True)
    test_data = test_data.remove_columns("output")
    test_data = test_data.remove_columns("input")
    write_log("Data processed.", args)

    result_path = f"./data/{args.dataset}/proc_data/data/evaluation_results"

    def compute_metrics(eval_preds):
        pre, labels = eval_preds
        np.save(
            os.path.join(
                result_path,
                f"_{args.chunk_interval}_{args.data_type}_pre",
            ),
            pre,
        )
        np.save(
            os.path.join(
                result_path,
                f"_{args.chunk_interval}_{args.data_type}_label",
            ),
            labels,
        )
        auc = roc_auc_score(pre[1], pre[0])
        ll = log_loss(pre[1], pre[0])
        acc = accuracy_score(pre[1], pre[0] > 0.5)
        return {
            "auc": auc,
            "ll": ll,
            "acc": acc,
        }

    trainer = transformers.Trainer(
        model=model,
        train_dataset=test_data,
        eval_dataset=test_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=False,
            logging_steps=1,
            evaluation_strategy="epoch" if VAL_SET_SIZE > 0 else "no",
            save_strategy="epoch",
            eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
            save_steps=args.save_steps,
            output_dir=OUTPUT_DIR,
            save_total_limit=30,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.wandb else [],
            ignore_data_skip=args.ignore_data_skip,
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            model.llama_tokenizer, return_tensors="pt", padding="longest"
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.llama_model.config.use_cache = False

    write_log("Evaluate on the test set...", args)
    write_log(trainer.evaluate(eval_dataset=test_data), args)


if __name__ == "__main__":
    args = get_args()
    main(args)
