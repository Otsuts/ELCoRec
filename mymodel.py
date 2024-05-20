import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import OrderedDict
from transformers import LlamaModel, LlamaForCausalLM
from transformers import LlamaTokenizer
from accelerate import hooks
from torch.nn.modules import module
from graph_dataset import (
    GraphDataset,
    GraphDataset_AZ_Toys,
    GraphDataset_ml_25m,
    GraphDataset_BookCrossing,
    Collator,
)
from torch.utils.data import DataLoader
from gat import GAT, GAT_BookCrossing

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


class LLM4Rec(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        load_in_8bit,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules,
        model_path,
        train_size=65536,
        K=15,
        set="train",
        dataset="ml-1m",
        chunk_interval="0:-1",
        freeze_emb_table=True,
    ):
        super(LLM4Rec, self).__init__()

        self.input_dim, self.output_dim = input_dim, output_dim

        print(f"Initializing language decoder ...")

        self.llama_model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map="auto",
        )
        if load_in_8bit:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
        if use_lora:
            # add the lora module
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            print("Lora used")
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            model_path, add_eos_token=True
        )
        self.llama_tokenizer.pad_token = (
            0  # This is the <unk> token, instead of eos token(id=1, <\s>)
        )
        self.llama_tokenizer.padding_side = "right"
        print("Language decoder initialized.")
        if dataset == "ml-1m":
            self.gat = GAT(
                self.input_dim,
                self.input_dim,
                num_feat=18889,
                user_fields=5,
                item_fields=3,
                feature_fields=1,
            )
            self.train_dataset = GraphDataset(
                "ml-1m",
                f"{set}",
                K,
                train_size=train_size,
                chunk_interval=chunk_interval,
            )
        elif dataset == "ml-25m":
            self.gat = GAT(
                self.input_dim,
                self.input_dim,
                num_feat=281000,
                user_fields=1,
                item_fields=3,
                feature_fields=1,
            )
            self.train_dataset = GraphDataset_ml_25m(
                "ml-25m",
                f"{set}",
                K,
                train_size=train_size,
                chunk_interval=chunk_interval,
            )
        elif dataset == "BookCrossing":
            self.gat = GAT_BookCrossing(
                self.input_dim,
                self.input_dim,
                num_feat=930000,
                user_fields=3,
                item_fields=2,
            )
            self.train_dataset = GraphDataset_BookCrossing(
                "BookCrossing",
                f"{set}",
                K,
                train_size=train_size,
                chunk_interval=chunk_interval,
            )

        elif dataset == "AZ-Toys":
            self.gat = GAT(
                self.input_dim,
                self.input_dim,
                num_feat=380000,
                user_fields=1,
                item_fields=3,
                feature_fields=1,
            )
            self.train_dataset = GraphDataset_AZ_Toys(
                "AZ-Toys",
                f"{set}",
                K,
                train_size=train_size,
                chunk_interval=chunk_interval,
            )

        self.gat.load_state_dict(torch.load(f"trained_models/{dataset}/GATGAARA.pth"))
        self.collator = Collator()

        if freeze_emb_table:
            for name, param in self.gat.named_parameters():
                param.requires_grad = False
            self.gat = self.gat.eval()
            print("Freeze item embedding table")

        self.embedding_proj = nn.Linear(
            self.input_dim, self.llama_model.config.hidden_size
        )

    def forward(self, *args, **kwargs):
        input_ids, labels, attention_mask, gat_idx = (
            kwargs["input_ids"],
            kwargs["labels"],
            kwargs["attention_mask"],
            kwargs["gat_input"],
        )
        user_feat, g = self.collator.collate_llm(
            self.train_dataset.__getitem__(gat_idx.item())
        )
        bs, seq_lenth = input_ids.shape[0], input_ids.shape[1]
        unk_token_id = self.llama_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(
            input_ids == unk_token_id
        )  # shape [Num of index, bs]
        assert replaced_idx.shape[0] == 1
        remain_idx = torch.nonzero(input_ids != unk_token_id)
        prompt_embeds = self.llama_model.base_model.model.model.embed_tokens(
            input_ids[
                remain_idx[:, 0],
                remain_idx[:, 1],
            ]
        )  # [bs, seq_lenth, embedding_size]
        x_emb = torch.zeros([bs, seq_lenth, 5120]).to(prompt_embeds.device)
        user_feat = user_feat.to(prompt_embeds.device)
        g = g.to(prompt_embeds.device)
        user_emb, target_item_emb = self.gat.get_emb(user_feat, g)

        item_embedding = self.embedding_proj(target_item_emb).view(-1, self.output_dim)

        x_emb[replaced_idx[:, 0], replaced_idx[:, 1], :] = item_embedding
        x_emb[remain_idx[:, 0], remain_idx[:, 1], :] = prompt_embeds
        assert (
            attention_mask.shape[0] == x_emb.shape[0]
            and attention_mask.shape[1] == x_emb.shape[1]
        )
        return self.llama_model.forward(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
