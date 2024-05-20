import sys

sys.path.append("../../")
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import argparse
from load_prompt_ml1m import zero_shot_get_prompt
import json


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--data_type", type=str, default="retatt_sampled")
parser.add_argument(
    "--temp_type",
    type=str,
    default="extra_embedding",
    choices=[
        "simple",
        "synthesis",
        "with_likes",
        "with_likes_simple",
        "extra_embedding",
        "extra_embedding_only",
        "mine_simple",
        "mine_complex",
        "RRAP",
    ],
)
parser.add_argument("--set", type=str, default="train", help="train/valid/test")
parser.add_argument("--train_size", type=int, default=65536)

args = parser.parse_args()
print(args)

temp_type_check = {
    "simple": [
        "rellaseq_sampled",
        "rethigh_sampled",
        "retatt_sampled",
        "retseq_sampled",
    ],
    "synthesis": ["rellasubseq_sampled", "retsubseq_sampled", "retattseq_sampled"],
    "with_likes": [
        "rellasubseqratinglike_sampled",
        "retsubseqratinglike_sampled",
        "retattseqratinglike_sampled",
    ],
    "with_likes_simple": [
        "rellaseqratinglike_sampled",
    ],
    "extra_embedding": [
        "rellasubseq_sampled",
        "retsubseq_sampled",
        "retattseq_sampled",
    ],
    "extra_embedding_only": [
        "rellasubseq_sampled",
        "retsubseq_sampled",
        "retattseq_sampled",
    ],
    "mine_simple": ["mine"],
    "mine_complex": ["mine"],
    "RRAP": ["GAARA"],
    "RRAP_simpled": ["GAARA"],
}
assert (
    args.data_type in temp_type_check[args.temp_type]
), "Wrong template type and data type combination"
temp_type = args.temp_type

target_dir = "../../data/ml-1m/proc_data/data/intermediate_data"
embedding_dir = "../../data/ml-1m/PLM_data"

if not os.path.exists(
    os.path.join(target_dir, f"{args.set}_{args.data_type}_{args.K}.parquet.gz")
):
    # sort by genre
    # get retrieval results first
    df_data = pd.read_parquet(
        os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz")
    )
    if args.data_type == "rethigh_sampled":
        pass
    elif "rellaseq" in args.data_type or "rellasubseq" in args.data_type:
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
        )
    elif args.data_type == "mine" or args.data_type == "GAARA":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
        )
    elif "retsubseq" in args.data_type:
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
        )
        print(df_data.columns)
        genre_list = [
            "Romance",
            "Horror",
            "Crime",
            "Western",
            "Mystery",
            "Film-Noir",
            "War",
            "Musical",
            "Thriller",
            "Fantasy",
            "Action",
            "Adventure",
            "Comedy",
            "Drama",
            "Sci-Fi",
            "Documentary",
            "Children's",
            "Animation",
        ]
        sorted_indice = np.load(os.path.join(embedding_dir, f"genre_indice.npy"))
        # 维护movie的genre字典
        movie_id_dict = json.load(
            open("../../data/ml-1m/proc_data/data/intermediate_data/ctr-meta.json")
        )["feature_dict"][
            "Movie ID"
        ]  # row to encode
        print(len(movie_id_dict))
        movie_to_genre = {}
        movie_data = []
        movie_fields = ["Movie ID", "Movie title", "Movie genre"]
        for line in open(
            os.path.join("../../data/ml-1m/raw_data", "movies.dat"),
            "r",
            encoding="ISO-8859-1",
        ).readlines():
            ele = line.strip().split("::")
            movie_id = ele[0].strip()
            movie_title = ele[1].strip()
            movie_genre = ele[2].strip().split("|")[0]
            movie_data.append([movie_id, movie_title, movie_genre])

        df_movie = pd.DataFrame(movie_data, columns=movie_fields)

        for _, row in df_movie.iterrows():
            if str(row["Movie ID"]) in movie_id_dict:
                movie_to_genre[str(row["Movie ID"])] = row[
                    "Movie genre"
                ]  # raw_data->movie genre

        genre_history = []
        genre_ratings = []
        for idx, row in tqdm(df_data.iterrows()):
            target_id, target_genre, hist = (
                int(row["Movie ID"]),
                row["Movie genre"],
                list(row["user_hist"]),
            )
            genre_idx = genre_list.index(target_genre)
            # 跟target item相关的genre的index
            cur_indice = list(sorted_indice[genre_idx])

            genre_hist = sorted(
                hist,
                key=lambda x: cur_indice.index(genre_list.index(movie_to_genre[x])),
            )
            sorted_idx = np.array(
                list(map(lambda x: hist.index(x), genre_hist)), dtype=np.int64
            )
            genre_rating = row["hist_rating"][sorted_idx]
            genre_history.append(genre_hist)
            genre_ratings.append(genre_rating)
        df_data["user_hist"] = genre_history
        df_data["hist_rating"] = genre_ratings
    elif args.data_type == "retatt_sampled":
        att_scores = np.load(f"../aux_data/att_scores_{args.set}.npy")
        att_idxes = np.argsort(-att_scores, axis=1)
        att_historys = []
        att_ratings = []
        for idx, row in tqdm(df_data.iterrows()):
            hist, hist_ratings = row["user_hist"], row["hist_rating"]
            att_idx = att_idxes[idx][: len(hist)]

            att_historys.append(hist[att_idx])
            att_ratings.append(hist_ratings[att_idx])

        df_data["user_hist"] = att_historys
        df_data["hist_rating"] = att_ratings
    elif args.data_type == "retattseq_sampled":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
        )
        att_scores = np.load(f"../aux_data/att_scores_{args.set}.npy")
        att_idxes = np.argsort(-att_scores, axis=1)
        att_historys = []
        att_ratings = []
        for idx, row in tqdm(df_data.iterrows()):
            hist, hist_ratings = row["user_hist"], row["hist_rating"]
            att_idx = att_idxes[idx][: len(hist)]

            att_historys.append(hist[att_idx])
            att_ratings.append(hist_ratings[att_idx])

        df_data["user_hist"] = att_historys
        df_data["hist_rating"] = att_ratings
    elif args.data_type == "retseq_sampled":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz")
        )
        genre_list = [
            "Romance",
            "Horror",
            "Crime",
            "Western",
            "Mystery",
            "Film-Noir",
            "War",
            "Musical",
            "Thriller",
            "Fantasy",
            "Action",
            "Adventure",
            "Comedy",
            "Drama",
            "Sci-Fi",
            "Documentary",
            "Children's",
            "Animation",
        ]
        sorted_indice = np.load(os.path.join(embedding_dir, f"genre_indice.npy"))
        # 维护movie的genre字典
        movie_id_dict = json.load(
            open("../../data/ml-1m/proc_data/data/intermediate_data/ctr-meta.json")
        )["feature_dict"][
            "Movie ID"
        ]  # row to encode
        print(len(movie_id_dict))
        movie_to_genre = {}
        movie_data = []
        movie_fields = ["Movie ID", "Movie title", "Movie genre"]
        for line in open(
            os.path.join("../../data/ml-1m/raw_data", "movies.dat"),
            "r",
            encoding="ISO-8859-1",
        ).readlines():
            ele = line.strip().split("::")
            movie_id = ele[0].strip()
            movie_title = ele[1].strip()
            movie_genre = ele[2].strip().split("|")[0]
            movie_data.append([movie_id, movie_title, movie_genre])

        df_movie = pd.DataFrame(movie_data, columns=movie_fields)

        for _, row in df_movie.iterrows():
            if str(row["Movie ID"]) in movie_id_dict:
                movie_to_genre[str(row["Movie ID"])] = row[
                    "Movie genre"
                ]  # raw_data->movie genre

        genre_history = []
        genre_ratings = []
        for idx, row in tqdm(df_data.iterrows()):
            target_id, target_genre, hist = (
                int(row["Movie ID"]),
                row["Movie genre"],
                list(row["user_hist"]),
            )
            genre_idx = genre_list.index(target_genre)
            # 跟target item相关的genre的index
            cur_indice = list(sorted_indice[genre_idx])

            genre_hist = sorted(
                hist,
                key=lambda x: cur_indice.index(genre_list.index(movie_to_genre[x])),
            )
            sorted_idx = np.array(
                list(map(lambda x: hist.index(x), genre_hist)), dtype=np.int64
            )
            genre_rating = row["hist_rating"][sorted_idx]
            genre_history.append(genre_hist)
            genre_ratings.append(genre_rating)
        df_data["user_hist"] = genre_history
        df_data["hist_rating"] = genre_ratings
    else:
        raise NotImplementedError
    df_data.to_parquet(
        os.path.join(target_dir, f"{args.set}_{args.data_type}_{args.K}.parquet.gz")
    )

DATA_DIR = f"../../data/{args.dataset}/proc_data/"

assert args.set in ["train", "test"]

fp = os.path.join(DATA_DIR, "data")
os.makedirs(fp, exist_ok=True)
fp = os.path.join(fp, args.set)
os.makedirs(fp, exist_ok=True)
file_name = "_".join(
    [args.set, f"{args.data_type}_{args.temp_type}_{args.K}", f"{args.train_size}"]
)
fp = os.path.join(fp, file_name + ".json")
ori_data_fp = os.path.join(
    target_dir, f"{args.set}_{args.data_type}_{args.K}.parquet.gz"
)
df = pd.read_parquet(ori_data_fp)
if args.train_size > 0:
    print(f"Sampling training dataset to size {args.train_size}")
    df = df.sample(args.train_size, replace=False, random_state=42)
    df = df.reset_index()
df.to_parquet(
    os.path.join(
        target_dir, f"{args.set}_{args.data_type}_{args.K}_{args.train_size}.parquet.gz"
    )
)

if args.dataset == "ml-1m":
    msg_iter = zero_shot_get_prompt(
        K=args.K,
        istrain=args.set,
        temp_type=temp_type,
        data_dir=target_dir,
        fp=f"{args.set}_{args.data_type}_{args.K}_{args.train_size}.parquet.gz",
        rating_use_likes="ratinglike" in args.data_type,
    )

data_list = []
for msg, idx in zip(msg_iter, df.index):
    encoded_idx = None
    if isinstance(msg, tuple):
        msg, encoded_idx = msg[0], msg[1]
    labels = df.loc[idx, "labels"]
    data_dict = {}
    data_dict["input"] = msg
    data_dict["output"] = "Yes." if int(labels) == 1 else "No."
    if encoded_idx:
        data_dict["encoded_idx"] = encoded_idx
    data_list.append(data_dict)

assert len(data_list) == len(df.index)

json.dump(data_list, open(fp, "w"), indent=4)
