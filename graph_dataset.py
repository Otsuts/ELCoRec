import dgl
import os
import pandas as pd
import json
import gc
from torch.utils.data import Dataset, DataLoader
import torch
import networkx as nx
import matplotlib.pyplot as plt


class GraphDataset(Dataset):
    def __init__(
        self, dataset, set, K, train_size=32768, chunk_interval="0:-1", seed=42
    ):
        super().__init__()
        data_dir = (
            f"data/{dataset}/proc_data/data/intermediate_data/{set}_ts.parquet.gz"
        )

        ctr_meta = json.load(
            open(f"data/{dataset}/proc_data/data/intermediate_data/ctr-meta.json", "r")
        )
        (
            feature_offset,
            rawitem2encode,
            genre2encode,
            rawuser2encode,
            gender2encode,
            age2encode,
            job2encode,
            zipcode2encode,
        ) = (
            ctr_meta["feature_offset"],
            ctr_meta["feature_dict"]["Movie ID"],
            ctr_meta["feature_dict"]["Movie genre"],
            ctr_meta["feature_dict"]["User ID"],
            ctr_meta["feature_dict"]["Gender"],
            ctr_meta["feature_dict"]["Age"],
            ctr_meta["feature_dict"]["Job"],
            ctr_meta["feature_dict"]["Zipcode"],
        )

        movie_info = pd.read_parquet(
            f"data/{dataset}/proc_data/data/intermediate_data/movies.parquet.gz"
        )
        user_info = pd.read_parquet(
            f"data/{dataset}/proc_data/data/intermediate_data/users.parquet.gz"
        )

        self.id2user = {}  # Mapping from user id to encoded user feature

        for idx, row in user_info.iterrows():
            try:
                self.id2user[rawuser2encode[row["User ID"]]] = [
                    rawuser2encode[row["User ID"]],
                    gender2encode[row["Gender"]] + 6042,
                    age2encode[row["Age"]] + 6049,
                    job2encode[row["Job"]] + 6070,
                    zipcode2encode[row["Zipcode"]] + 9509,
                ]
            except:
                pass

        self.id2genre = {}  # Mapping from encoded movie id to encoded movie genre
        for idx, row in movie_info.iterrows():
            try:
                self.id2genre[rawitem2encode[row["Movie ID"]]] = (
                    genre2encode[row["Movie genre"]] + 16912
                )
            except:
                pass
        self.df_file = pd.read_parquet(data_dir)[
            [
                "User ID",
                "Movie ID",
                "rating",
                "labels",
                "user_hist",
                "hist_rating",
                "timestamp",
                "hist_timestamp",
            ]
        ]

        self.df_file["User ID"] = self.df_file["User ID"].apply(
            lambda x: rawuser2encode[str(x)]
        )
        self.df_file["Movie ID"] = self.df_file["Movie ID"].apply(
            lambda x: rawitem2encode[str(x)]
        )
        self.df_file["user_hist"] = self.df_file["user_hist"].apply(lambda x: x[-K:])
        self.df_file["hist_rating"] = self.df_file["hist_rating"].apply(
            lambda x: x[-K:]
        )
        self.df_file["hist_timestamp"] = self.df_file["hist_timestamp"].apply(
            lambda x: x[-K:]
        )
        if set == "train" and train_size > 0:
            self.df_file = self.df_file.sample(
                train_size, replace=False, random_state=seed
            )
            self.df_file = self.df_file.reset_index()
        if set == "test":
            start_idx, end_idx = int(chunk_interval.split(":")[0]), int(
                chunk_interval.split(":")[1]
            )
            print(f"getting sample from {start_idx} to {end_idx}")
            self.df_file = self.df_file.iloc[start_idx:end_idx].reset_index()
        print(f"dataset lenth: {len(self.df_file)}")

        self.rawitem2encode = rawitem2encode
        del (
            ctr_meta,
            feature_offset,
            rawitem2encode,
            genre2encode,
            rawuser2encode,
            gender2encode,
            age2encode,
            job2encode,
            zipcode2encode,
        )

    def __len__(self):
        return len(self.df_file)

    def build_graph(self, item_list_ids):
        item_len = len(item_list_ids)
        feat_num = len(set(map(lambda x: self.id2genre[x], item_list_ids)))
        item_nodes = torch.arange(start=0, end=item_len)

        feat_nodes = list(range(item_len, item_len + feat_num))
        # Use dgl to build graph
        i2f_source = item_nodes
        hist_genre = sorted(
            list(
                map(
                    lambda x: self.id2genre[x] - 16912 + item_len,
                    item_list_ids,
                )
            )
        )
        hist_genre_reencode = list(set(hist_genre))
        i2f_target = torch.tensor(
            list(map(lambda x: feat_nodes[hist_genre_reencode.index(x)], hist_genre))
        )
        edge_src = i2f_source
        edge_dst = i2f_target - item_len
        u2i_src = [0] * len(item_nodes)
        u2i_dst = item_nodes
        g = dgl.heterograph(
            {
                ("Item", "belongto", "Feature"): (edge_src, edge_dst),
                ("Feature", "hasinstance", "Item"): (edge_dst, edge_src),
                ("User", "interacted", "Item"): (u2i_src, u2i_dst),
                ("Item", "clickby", "User"): (u2i_dst, u2i_src),
            }
        )
        return g

    def timestamp_transformation(self, hist_timestamp):
        result = torch.ones(len(hist_timestamp) + 1, 32)
        target_timestamp = [self.target_timestamp]
        target_timestamp.extend(hist_timestamp)  # Total timestamp
        pos = torch.tensor(
            list(
                map(
                    lambda x: self.target_timestamp - x,
                    target_timestamp,
                )
            )
        ).unsqueeze(1)
        i = torch.arange(32)
        div = 10000 ** (i / 32)
        term = pos / div
        result[:, 0::2] = torch.sin(term[:, 0::2])
        result[:, 1::2] = torch.cos(term[:, 0::2])
        return result

    def __getitem__(self, index):
        data = self.df_file.iloc[index]
        label = data["labels"]
        hist_rating = data["hist_rating"]
        self.target_timestamp = data["timestamp"]

        user_feature = list(map(lambda x: self.id2user[x], [data["User ID"]]))
        item_feature = list(map(lambda x: [x, self.id2genre[x]], [data["Movie ID"]]))
        hist_id = list(
            map(
                lambda x: self.rawitem2encode[x],
                data["user_hist"],  # History ID
            )
        )

        # Build graph
        g = self.build_graph([item_feature[0][0]] + hist_id)
        # nx.draw(g.to_networkx(), with_labels=True)  # 可视化图
        # plt.savefig("fig.png")
        # assert 0
        g.nodes["Item"].data["id"] = torch.tensor([item_feature[0][0]] + hist_id)
        g.nodes["Item"].data["rating"] = torch.tensor([0] + list(hist_rating)) + 16939
        g.nodes["Item"].data["ts_emb"] = self.timestamp_transformation(
            data["hist_timestamp"]
        )
        g.nodes["Item"].data["is_target"] = torch.where(
            g.nodes["Item"].data["id"] == data["Movie ID"], 1, 0
        )
        g.nodes["Feature"].data["id"] = torch.tensor(
            list(set(map(lambda x: self.id2genre[x], [item_feature[0][0]] + hist_id)))
        )
        user_feature = torch.tensor(user_feature).squeeze()
        if len(user_feature.shape) == 1:
            user_feature = user_feature.unsqueeze(0)
        return user_feature, label, g


class GraphDataset_ml_25m(Dataset):
    def __init__(
        self, dataset, set, K, train_size=32768, chunk_interval="0:-1", seed=42
    ):
        super().__init__()
        data_dir = (
            f"data/{dataset}/proc_data/data/intermediate_data/{set}_ts.parquet.gz"
        )

        ctr_meta = json.load(
            open(f"data/{dataset}/proc_data/data/intermediate_data/ctr-meta.json", "r")
        )
        (
            feature_offset,
            rawitem2encode,
            genre2encode,
            rawuser2encode,
        ) = (
            ctr_meta["feature_offset"],
            ctr_meta["feature_dict"]["Movie ID"],
            ctr_meta["feature_dict"]["Movie genre"],
            ctr_meta["feature_dict"]["User ID"],
        )
        movie_info = pd.read_parquet(
            f"data/{dataset}/proc_data/data/intermediate_data/movies.parquet.gz"
        )

        self.id2genre = {}  # Mapping from encoded movie id to encoded movie genre
        for idx, row in movie_info.iterrows():
            try:
                self.id2genre[rawitem2encode[str(row["Movie ID"])]] = (
                    genre2encode[row["Movie genre"]] + 280546
                )
            except:
                pass
        self.df_file = pd.read_parquet(data_dir)[
            [
                "User ID",
                "Movie ID",
                "rating",
                "labels",
                "user_hist",
                "hist_rating",
                "timestamp",
                "hist_timestamp",
            ]
        ]

        # self.df_file["User ID"] = self.df_file["User ID"].apply(
        #     lambda x: rawuser2encode[str(x)]
        # )
        # self.df_file["Movie ID"] = self.df_file["Movie ID"].apply(
        #     lambda x: rawitem2encode[str(x)]
        # )
        self.df_file["user_hist"] = self.df_file["user_hist"].apply(lambda x: x[-K:])
        self.df_file["rating"] = self.df_file["rating"].apply(lambda x: int(x * 2))
        self.df_file["hist_rating"] = self.df_file["hist_rating"].apply(
            lambda x: x[-K:]
        )
        self.df_file["hist_rating"] = self.df_file["hist_rating"].apply(
            lambda x: [int(y * 2) for y in x]
        )
        self.df_file["hist_timestamp"] = self.df_file["hist_timestamp"].apply(
            lambda x: x[-K:]
        )
        if set == "train" and train_size > 0:
            self.df_file = self.df_file.sample(
                train_size, replace=False, random_state=seed
            )
            self.df_file = self.df_file.reset_index()
        if set == "test":
            start_idx, end_idx = int(chunk_interval.split(":")[0]), int(
                chunk_interval.split(":")[1]
            )
            print(f"getting sample from {start_idx} to {end_idx}")
            self.df_file = self.df_file.iloc[start_idx:end_idx].reset_index()
        print(f"dataset lenth: {len(self.df_file)}")

        self.rawitem2encode = rawitem2encode
        del (
            ctr_meta,
            feature_offset,
            rawitem2encode,
            genre2encode,
            rawuser2encode,
        )

    def __len__(self):
        return len(self.df_file)

    def build_graph(self, item_list_ids):
        item_len = len(item_list_ids)
        feat_num = len(set(map(lambda x: self.id2genre[x], item_list_ids)))
        item_nodes = torch.arange(start=0, end=item_len)

        feat_nodes = list(range(item_len, item_len + feat_num))
        # Use dgl to build graph
        i2f_source = item_nodes
        hist_genre = sorted(
            list(
                map(
                    lambda x: self.id2genre[x] - 280546 + item_len,
                    item_list_ids,
                )
            )
        )
        hist_genre_reencode = list(set(hist_genre))
        i2f_target = torch.tensor(
            list(map(lambda x: feat_nodes[hist_genre_reencode.index(x)], hist_genre))
        )
        edge_src = i2f_source
        edge_dst = i2f_target - item_len
        u2i_src = [0] * len(item_nodes)
        u2i_dst = item_nodes
        g = dgl.heterograph(
            {
                ("Item", "belongto", "Feature"): (edge_src, edge_dst),
                ("Feature", "hasinstance", "Item"): (edge_dst, edge_src),
                ("User", "interacted", "Item"): (u2i_src, u2i_dst),
                ("Item", "clickby", "User"): (u2i_dst, u2i_src),
            }
        )
        return g

    def timestamp_transformation(self, hist_timestamp):
        result = torch.ones(len(hist_timestamp) + 1, 32)
        target_timestamp = [self.target_timestamp]
        target_timestamp.extend(hist_timestamp)  # Total timestamp
        pos = torch.tensor(
            list(
                map(
                    lambda x: self.target_timestamp - x,
                    target_timestamp,
                )
            )
        ).unsqueeze(1)
        i = torch.arange(32)
        div = 10000 ** (i / 32)
        term = pos / div
        result[:, 0::2] = torch.sin(term[:, 0::2])
        result[:, 1::2] = torch.cos(term[:, 0::2])
        return result

    def __getitem__(self, index):
        data = self.df_file.iloc[index]
        label = data["labels"]
        hist_rating = data["hist_rating"]
        self.target_timestamp = data["timestamp"]

        user_feature = list(map(lambda x: x, [data["User ID"]]))
        item_feature = list(map(lambda x: [x, self.id2genre[x]], [data["Movie ID"]]))
        hist_id = data["user_hist"].tolist()
        # Build graph
        g = self.build_graph([item_feature[0][0]] + hist_id)
        # nx.draw(g.to_networkx(), with_labels=True)  # 可视化图
        # plt.savefig("fig.png")
        g.nodes["Item"].data["id"] = torch.tensor([item_feature[0][0]] + hist_id)
        g.nodes["Item"].data["rating"] = torch.tensor([0] + list(hist_rating)) + 280566
        g.nodes["Item"].data["ts_emb"] = self.timestamp_transformation(
            data["hist_timestamp"]
        )
        g.nodes["Item"].data["is_target"] = torch.where(
            g.nodes["Item"].data["id"] == data["Movie ID"], 1, 0
        )
        g.nodes["Feature"].data["id"] = torch.tensor(
            list(set(map(lambda x: self.id2genre[x], [item_feature[0][0]] + hist_id)))
        )
        user_feature = torch.tensor(user_feature).squeeze()
        if len(user_feature.shape) == 1:
            user_feature = user_feature.unsqueeze(0)
        return user_feature, label, g


class GraphDataset_BookCrossing(Dataset):
    def __init__(
        self, dataset, set, K, train_size=32768, chunk_interval="0:-1", seed=42
    ):
        super().__init__()
        data_dir = f"data/{dataset}/proc_data/data/intermediate_data/{set}.parquet.gz"

        ctr_meta = json.load(
            open(f"data/{dataset}/proc_data/data/intermediate_data/ctr-meta.json", "r")
        )
        (
            feature_offset,
            rawitem2encode,
            author2encode,
            year2encode,
            publisher2encode,
            rawuser2encode,
            location2encode,
            age2encode,
        ) = (
            ctr_meta["feature_offset"],
            ctr_meta["feature_dict"]["ISBN"],
            ctr_meta["feature_dict"]["Author"],
            ctr_meta["feature_dict"]["Publication year"],
            ctr_meta["feature_dict"]["Publisher"],
            ctr_meta["feature_dict"]["User ID"],
            ctr_meta["feature_dict"]["Location"],
            ctr_meta["feature_dict"]["Age"],
        )

        book_info = pd.read_parquet(
            f"data/{dataset}/proc_data/data/intermediate_data/books.parquet.gz"
        )
        user_info = pd.read_parquet(
            f"data/{dataset}/proc_data/data/intermediate_data/users.parquet.gz"
        )

        self.id2user = {}  # Mapping from user id to encoded user feature

        for idx, row in user_info.iterrows():
            try:
                self.id2user[rawuser2encode[row["User ID"]]] = [
                    rawuser2encode[row["User ID"]],
                    location2encode[row["Location"]] + 278858,
                    age2encode[row["Age"]] + 279780,
                ]
            except:
                pass

        self.id2genre = {}  # Mapping from encoded movie id to encoded movie genre
        for idx, row in book_info.iterrows():
            try:
                self.id2genre[rawitem2encode[row["ISBN"]]] = (
                    author2encode[row["Author"]] + 793318
                )

            except:
                pass
        self.df_file = pd.read_parquet(data_dir)[
            [
                "User ID",
                "ISBN",
                "rating",
                "labels",
                "user_hist",
                "hist_rating",
            ]
        ]

        self.df_file["User ID"] = self.df_file["User ID"].apply(
            lambda x: rawuser2encode[str(x)]
        )
        self.df_file["ISBN"] = self.df_file["ISBN"].apply(
            lambda x: rawitem2encode[str(x)]
        )
        self.df_file["user_hist"] = self.df_file["user_hist"].apply(lambda x: x[-K:])
        self.df_file["hist_rating"] = self.df_file["hist_rating"].apply(
            lambda x: x[-K:]
        )
        if set == "train" and train_size > 0:
            self.df_file = self.df_file.sample(
                train_size, replace=False, random_state=seed
            )
            self.df_file = self.df_file.reset_index()
        if set == "test":
            start_idx, end_idx = int(chunk_interval.split(":")[0]), int(
                chunk_interval.split(":")[1]
            )
            print(f"getting sample from {start_idx} to {end_idx}")
            self.df_file = self.df_file.iloc[start_idx:end_idx].reset_index()
        print(f"dataset lenth: {len(self.df_file)}")

        self.rawitem2encode = rawitem2encode
        del (
            feature_offset,
            rawitem2encode,
            author2encode,
            year2encode,
            publisher2encode,
            rawuser2encode,
            location2encode,
            age2encode,
        )

    def __len__(self):
        return len(self.df_file)

    def build_graph(self, item_list_ids):
        item_len = len(item_list_ids)
        feat_num = len(set(map(lambda x: self.id2genre[x], item_list_ids)))
        item_nodes = torch.arange(start=0, end=item_len)

        feat_nodes = list(range(item_len, item_len + feat_num))
        # Use dgl to build graph
        i2f_source = item_nodes
        hist_genre = sorted(
            list(
                map(
                    lambda x: self.id2genre[x] - 793318 + item_len,
                    item_list_ids,
                )
            )
        )
        hist_genre_reencode = list(set(hist_genre))
        i2f_target = torch.tensor(
            list(map(lambda x: feat_nodes[hist_genre_reencode.index(x)], hist_genre))
        )
        edge_src = i2f_source
        edge_dst = i2f_target - item_len
        u2i_src = [0] * len(item_nodes)
        u2i_dst = item_nodes
        g = dgl.heterograph(
            {
                ("Item", "belongto", "Feature"): (edge_src, edge_dst),
                ("Feature", "hasinstance", "Item"): (edge_dst, edge_src),
                ("User", "interacted", "Item"): (u2i_src, u2i_dst),
                ("Item", "clickby", "User"): (u2i_dst, u2i_src),
            }
        )
        return g

    def timestamp_transformation(self, hist_timestamp):
        result = torch.ones(len(hist_timestamp) + 1, 32)
        target_timestamp = [self.target_timestamp]
        target_timestamp.extend(hist_timestamp)  # Total timestamp
        pos = torch.tensor(
            list(
                map(
                    lambda x: self.target_timestamp - x,
                    target_timestamp,
                )
            )
        ).unsqueeze(1)
        i = torch.arange(32)
        div = 10000 ** (i / 32)
        term = pos / div
        result[:, 0::2] = torch.sin(term[:, 0::2])
        result[:, 1::2] = torch.cos(term[:, 0::2])
        return result

    def __getitem__(self, index):
        data = self.df_file.iloc[index]
        label = data["labels"]
        hist_rating = data["hist_rating"]

        user_feature = list(map(lambda x: self.id2user[x], [data["User ID"]]))
        item_feature = list(
            map(
                lambda x: [x, self.id2genre[x]],
                [data["ISBN"]],
            )
        )
        hist_id = list(
            map(
                lambda x: self.rawitem2encode[x],
                data["user_hist"],  # History ID
            )
        )

        # Build graph
        g = self.build_graph([item_feature[0][0]] + hist_id)
        # nx.draw(g.to_networkx(), with_labels=True)  # 可视化图
        # plt.savefig("fig.png")
        # assert 0
        g.nodes["Item"].data["id"] = torch.tensor([item_feature[0][0]] + hist_id)
        g.nodes["Item"].data["rating"] = torch.tensor([0] + list(hist_rating)) + 16939
        # g.nodes["Item"].data["ts_emb"] = self.timestamp_transformation(
        #     data["hist_timestamp"]
        # )
        # g.nodes["Item"].data["feat"] = torch.cat(
        #     (
        #         torch.tensor(item_feature),
        #         torch.tensor(
        #             list(
        #                 map(
        #                     lambda x: [
        #                         x,
        #                         self.id2genre[x][0],
        #                         self.id2genre[x][1],
        #                         self.id2genre[x][2],
        #                     ],
        #                     hist_id,
        #                 )
        #             )
        #         ),
        #     )
        # )[:, 2:]
        g.nodes["Item"].data["is_target"] = torch.where(
            g.nodes["Item"].data["id"] == data["ISBN"], 1, 0
        )
        g.nodes["Feature"].data["id"] = torch.tensor(
            list(set(map(lambda x: self.id2genre[x], [item_feature[0][0]] + hist_id)))
        )
        user_feature = torch.tensor(user_feature).squeeze()
        if len(user_feature.shape) == 1:
            user_feature = user_feature.unsqueeze(0)
        return user_feature, label, g


class GraphDataset_AZ_Toys(Dataset):
    def __init__(
        self, dataset, split, K, train_size=32768, chunk_interval="0:-1", seed=42
    ):
        super().__init__()
        data_dir = (
            f"data/{dataset}/proc_data/data/intermediate_data/{split}_ts.parquet.gz"
        )

        ctr_meta = json.load(
            open(f"data/{dataset}/proc_data/data/intermediate_data/ctr-meta.json", "r")
        )
        (
            feature_offset,
            rawitem2encode,
            category2encode,
            brand2encode,
            rawuser2encode,
        ) = (
            ctr_meta["feature_offset"],
            ctr_meta["feature_dict"]["Item ID"],
            ctr_meta["feature_dict"]["Category"],
            ctr_meta["feature_dict"]["Brand"],
            ctr_meta["feature_dict"]["User ID"],
        )

        item_info = pd.read_parquet(
            f"data/{dataset}/proc_data/data/intermediate_data/items.parquet.gz"
        )

        self.id2category = {}  # Mapping from encoded movie id to encoded movie genre
        for idx, row in item_info.iterrows():
            try:
                self.id2category[rawitem2encode[row["Item ID"]]] = (
                    category2encode[row["Category"]] + 285864
                )

            except:
                pass

        self.id2brand = {}  # Mapping from encoded movie id to encoded movie genre
        for idx, row in item_info.iterrows():
            try:
                self.id2brand[rawitem2encode[row["Item ID"]]] = (
                    brand2encode[row["Brand"]] + 363024
                )

            except:
                pass
        self.df_file = pd.read_parquet(data_dir)[
            [
                "User ID",
                "Item ID",
                "rating",
                "labels",
                "user_hist",
                "hist_rating",
                "timestamp",
                "hist_timestamp",
            ]
        ]

        self.df_file["User ID"] = self.df_file["User ID"].apply(
            lambda x: rawuser2encode[str(x)]
        )
        self.df_file["Item ID"] = self.df_file["Item ID"].apply(
            lambda x: rawitem2encode[str(x)]
        )
        self.df_file["user_hist"] = self.df_file["user_hist"].apply(lambda x: x[-K:])
        self.df_file["hist_rating"] = self.df_file["hist_rating"].apply(
            lambda x: x[-K:]
        )
        self.df_file["hist_timestamp"] = self.df_file["hist_timestamp"].apply(
            lambda x: x[-K:]
        )
        if split == "train" and train_size > 0:
            self.df_file = self.df_file.sample(
                train_size, replace=False, random_state=seed
            )
            self.df_file = self.df_file.reset_index()
        if set == "test":
            start_idx, end_idx = int(chunk_interval.split(":")[0]), int(
                chunk_interval.split(":")[1]
            )
            print(f"getting sample from {start_idx} to {end_idx}")
            self.df_file = self.df_file.iloc[start_idx:end_idx].reset_index()
        print(f"dataset lenth: {len(self.df_file)}")

        self.rawitem2encode = rawitem2encode
        del (
            ctr_meta,
            feature_offset,
            rawitem2encode,
            category2encode,
            brand2encode,
            rawuser2encode,
        )

    def __len__(self):
        return len(self.df_file)

    def build_graph(self, item_list_ids):
        item_len = len(item_list_ids)
        category_num = len({self.id2category[x] for x in item_list_ids})

        # brand_num = len({self.id2brand[x] for x in item_list_ids})
        item_nodes = torch.arange(start=0, end=item_len)

        category_nodes = list(range(item_len, item_len + category_num))
        # brand_nodes = list(
        #     range(item_len + category_num, item_len + category_num + brand_num)
        # )
        # Use dgl to build graph
        i2f_source = item_nodes
        hist_category = sorted(
            [self.id2category[x] - 285864 + item_len for x in item_list_ids]
        )
        hist_category_reencode = list(set(hist_category))
        # hist_brand = sorted(
        #     list(
        #         map(
        #             lambda x: self.id2brand[x] - 363024 + item_len + category_num,
        #             item_list_ids,
        #         )
        #     )
        # )
        # hist_brand_reencode = list(set(hist_brand))
        i2f_target = torch.tensor(
            list(
                map(
                    lambda x: category_nodes[hist_category_reencode.index(x)],
                    hist_category,
                )
            )
            # + list(map(lambda x: brand_nodes[hist_brand_reencode.index(x)], hist_brand))
        )
        edge_src = i2f_source
        edge_dst = i2f_target - item_len
        u2i_src = [0] * len(item_nodes)
        u2i_dst = item_nodes
        g = dgl.heterograph(
            {
                ("Item", "belongto", "Feature"): (edge_src, edge_dst),
                ("Feature", "hasinstance", "Item"): (edge_dst, edge_src),
                ("User", "interacted", "Item"): (u2i_src, u2i_dst),
                ("Item", "clickby", "User"): (u2i_dst, u2i_src),
            }
        )
        return g

    def timestamp_transformation(self, hist_timestamp):
        result = torch.ones(len(hist_timestamp) + 1, 32)
        target_timestamp = [self.target_timestamp]
        target_timestamp.extend(hist_timestamp)  # Total timestamp
        pos = torch.tensor(
            list(
                map(
                    lambda x: self.target_timestamp - x,
                    target_timestamp,
                )
            )
        ).unsqueeze(1)
        i = torch.arange(32)
        div = 10000 ** (i / 32)
        term = pos / div
        result[:, 0::2] = torch.sin(term[:, 0::2])
        result[:, 1::2] = torch.cos(term[:, 0::2])
        return result

    def __getitem__(self, index):
        data = self.df_file.iloc[index]
        label = data["labels"]
        hist_rating = data["hist_rating"]
        self.target_timestamp = data["timestamp"]

        user_feature = list(map(lambda x: x, [data["User ID"]]))
        item_feature = list(
            map(
                lambda x: [x, self.id2category[x], self.id2brand[x]],
                [data["Item ID"]],
            )
        )
        hist_id = list(
            map(
                lambda x: self.rawitem2encode[x],
                data["user_hist"],  # History ID
            )
        )
        # Build graph
        g = self.build_graph([item_feature[0][0]] + hist_id)
        # nx.draw(g.to_networkx(), with_labels=True)  # 可视化图
        # plt.savefig("fig.png")
        # assert 0
        g.nodes["Item"].data["id"] = torch.tensor([item_feature[0][0]] + hist_id)
        g.nodes["Item"].data["rating"] = torch.tensor([0] + list(hist_rating)) + 371813
        g.nodes["Item"].data["ts_emb"] = self.timestamp_transformation(
            data["hist_timestamp"]
        )
        g.nodes["Item"].data["is_target"] = torch.where(
            g.nodes["Item"].data["id"] == data["Item ID"], 1, 0
        )
        g.nodes["Item"].data["is_target"][1:] = 0
        # print(g.nodes["Item"].data["is_target"])
        g.nodes["Feature"].data["id"] = torch.tensor(
            list(
                set(map(lambda x: self.id2category[x], [item_feature[0][0]] + hist_id))
            )
        )
        user_feature = torch.tensor(user_feature).squeeze()
        if len(user_feature.shape) == 1:
            user_feature = user_feature.unsqueeze(0)
        return user_feature, label, g


class Collator(object):
    def __init__(self) -> None:
        pass

    def collate(self, batch):
        batch_features, batch_labels, batch_graphs = map(list, zip(*batch))
        # nx.draw(dgl.batch(batch_graphs).to_networkx(), with_labels=True)  # 可视化图
        # plt.savefig("fig.png")
        # assert 0
        return (
            torch.stack(batch_features),
            torch.tensor(batch_labels).to(torch.float32),
            dgl.batch(batch_graphs),
        )

    def collate_llm(self, batch):
        batch_features, batch_labels, batch_graphs = batch
        return (
            torch.stack([batch_features]),
            dgl.batch([batch_graphs]),
        )


if __name__ == "__main__":
    dataset = "AZ-Toys"
    split = "train"
    mydataset = GraphDataset_AZ_Toys(dataset, split, 15)
    mydataset[[0]]
