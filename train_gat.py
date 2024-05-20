import torch
import torch.nn as nn

from graph_dataset import (
    GraphDataset,
    GraphDataset_AZ_Toys,
    GraphDataset_ml_25m,
    GraphDataset_BookCrossing,
    Collator,
)
from torch.utils.data import DataLoader
from gat import GAT, GAT_AZ_Toys, GAT_ml_25m, GAT_BookCrossing
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import argparse
import datetime
import os


def write_log(w, args):
    file_name = (
        args.log
        + "/"
        + args.dataset
        + "/"
        + args.model
        + "/"
        + datetime.date.today().strftime("%m%d")
        + f"_LR{args.lr}_WD{args.wd}.log"
    )
    if not os.path.exists(args.log + "/" + args.dataset + "/" + args.model + "/"):
        os.makedirs(args.log + "/" + args.dataset + "/" + args.model + "/")
    t0 = datetime.datetime.now().strftime("%H:%M:%S")
    info = "{} : {}".format(t0, w)
    print(info)
    if not args.test_mode:
        with open(file_name, "a") as f:
            f.write(info + "\n")


def get_args():
    parser = argparse.ArgumentParser()
    # strategy
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-1m", "AZ-Toys", "ml-25m", "BookCrossing"],
    )
    parser.add_argument("--log", type=str, default="./logs")
    parser.add_argument("--num_workers", type=int, default=8)
    # device
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    # model structure
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="GAT")
    # test
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--K", type=int, default=15)
    args = parser.parse_args()
    return args


def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


def evaluate_lp(model, data_loader, device):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for user_feat, label, g in data_loader:
            user_feat = user_feat.to(device)
            label = label.to(device)
            g = g.to(device)
            logits = model(user_feat, g)
            logits = logits.squeeze().detach().cpu().numpy().astype("float64")
            label = label.detach().cpu().numpy()
            predictions.append(logits)
            labels.append(label)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = ((predictions > 0.5) == labels).sum() / predictions.shape[0]
    auc = roc_auc_score(y_score=predictions, y_true=labels)
    logloss = log_loss(y_true=labels, y_pred=predictions, eps=1e-7, normalize=True)
    return auc, acc, logloss


def main(args):
    seed_all(42, 0)
    device = torch.device(
        f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu"
    )
    write_log(f"Device is {device}.", args)
    if args.dataset == "ml-1m":
        train_dataset = GraphDataset(args.dataset, "train", args.K, train_size=-1)
        valid_dataset = GraphDataset(args.dataset, "valid", args.K)
        test_dataset = GraphDataset(args.dataset, "test", args.K)
    elif args.dataset == "AZ-Toys":
        train_dataset = GraphDataset_AZ_Toys(
            args.dataset, "train", args.K, train_size=-1
        )
        valid_dataset = GraphDataset_AZ_Toys(args.dataset, "valid", args.K)
        test_dataset = GraphDataset_AZ_Toys(args.dataset, "test", args.K)
    elif args.dataset == "ml-25m":
        train_dataset = GraphDataset_ml_25m(
            args.dataset, "train", args.K, train_size=-1
        )
        valid_dataset = GraphDataset_ml_25m(args.dataset, "valid", args.K)
        test_dataset = GraphDataset_ml_25m(args.dataset, "test", args.K)
    elif args.dataset == "BookCrossing":
        train_dataset = GraphDataset_BookCrossing(
            args.dataset, "train", args.K, train_size=-1
        )
        valid_dataset = GraphDataset_BookCrossing(args.dataset, "test", args.K)
        test_dataset = GraphDataset_BookCrossing(args.dataset, "test", args.K)
    collator = Collator()
    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, collate_fn=collator.collate
    )
    valid_loader = DataLoader(
        valid_dataset, args.batch_size, shuffle=False, collate_fn=collator.collate
    )
    test_loader = DataLoader(
        test_dataset, args.batch_size, shuffle=False, collate_fn=collator.collate
    )
    if args.dataset == "ml-1m":
        model = GAT(
            args.embedding_size,
            args.embedding_size,
            num_feat=18889,
        )
    elif args.dataset == "AZ-Toys":
        model = GAT(
            args.embedding_size,
            args.embedding_size,
            num_feat=380000,
            user_fields=1,
            item_fields=3,
        )
    elif args.dataset == "ml-25m":
        model = GAT(
            args.embedding_size,
            args.embedding_size,
            num_feat=281000,
            user_fields=1,
            item_fields=3,
        )
    elif args.dataset == "BookCrossing":
        model = GAT_BookCrossing(
            args.embedding_size,
            args.embedding_size,
            num_feat=930000,
            user_fields=3,
            item_fields=2,
        )

    model.apply(weight_init)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_func = nn.BCELoss()
    write_log("Start training.", args)
    best_auc = 0.0
    kill_cnt = 0
    for epoch in range(args.epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f"Epoch: {epoch+1}/{args.epochs}")
            for user_feat, labels, g in train_loader:
                user_feat = user_feat.to(device)
                labels = labels.to(device)
                g = g.to(device)
                logits = model(user_feat, g)

                # Compute Loss
                loss = loss_func(logits, labels.float())
                train_loss.append(loss.item())

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.update()
                t.set_postfix({"Train loss:": f"{loss.item():.4f}"})
            train_loss = np.mean(train_loss)
            val_auc, val_acc, val_logloss = evaluate_lp(
                model, valid_loader, device=device
            )

            write_log(f"Epoch {epoch}: train loss: {train_loss:.6f}", args)
            write_log(
                f"val AUC: {val_auc:.6f},  val ACC: {val_acc:.6f}, val logloss: {val_logloss:.6f}",
                args,
            )
            # validate
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    f"./trained_models/{args.dataset}/_{args.model}_{args.lr}_{args.wd}.pth",
                )
                kill_cnt = 0
                print("saving model...")
            else:
                kill_cnt += 1
                if kill_cnt >= args.early_stop:
                    print(f"Early stop at epoch {epoch}")
                    write_log("best epoch: {}".format(best_epoch + 1), args)
                    break

    # test use the best model
    model.eval()
    model.load_state_dict(
        torch.load(
            f"trained_models/{args.dataset}/_{args.model}_{args.lr}_{args.wd}.pth"
        )
    )
    test_auc, test_acc, test_logloss = evaluate_lp(model, test_loader, device=device)
    write_log(f"***********Test Results:************", args)
    write_log(f"Test ACC: {test_acc:.4f}", args)
    write_log(f"Test Auc: {test_auc:.4f}\t Test logloss: {test_logloss:.4f}", args)


if __name__ == "__main__":
    args = get_args()
    main(args)
