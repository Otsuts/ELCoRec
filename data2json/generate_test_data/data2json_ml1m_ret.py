import sys

sys.path.append("../../")
import os
import json
from load_prompt_ml1m import zero_shot_get_prompt, zero_shot_ret_get_prompt
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--data_type", type=str, default="ret")
parser.add_argument("--set", type=str, default="test", help="train/valid/test")
parser.add_argument("--chunk_interval", type=str, default="0:-1")
parser.add_argument(
    "--temp_type",
    type=str,
    default="extra_embedding",
    choices=[
        "simple",
        "synthesis",
        "RRAP",
        "with_likes",
        "with_likes_simple",
        "extra_embedding",
        "extra_embedding_only",
        "mine_simple",
        "mine_complex",
    ],
)
args = parser.parse_args()
print(args)
temp_type_check = {
    "simple": ["rellaseq", "rethigh", "retatt", "retseq"],
    "synthesis": ["rellasubseq", "retsubseq", "retattseq"],
    "with_likes": [
        "rellasubseqratinglike",
        "retsubseqratinglike",
        "retattseqratinglike",
    ],
    "with_likes_simple": [
        "rellaseqratinglike",
    ],
    "extra_embedding": [
        "rellasubseq",
        "retsubseq",
        "retattseq",
    ],
    "extra_embedding_only": [
        "rellasubseq",
        "retsubseq",
        "retattseq",
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

# Get recent K items and sort them by genre
target_dir = "../../data/ml-1m/proc_data/data/intermediate_data"
embedding_dir = "../../data/ml-1m/PLM_data"

if not os.path.exists(
    os.path.join(target_dir, f"{args.set}_{args.data_type}_{args.K}.parquet.gz")
):
    if not os.path.exists(
        os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz")
    ):
        df_data = pd.read_parquet(os.path.join(target_dir, f"{args.set}.parquet.gz"))
        print(df_data.columns)
        sorted_indice = np.load(
            os.path.join(embedding_dir, f"zero_shot_average_indice.npy")
        )  # from raw id , [3952, 3952]
        print(sorted_indice.shape)

        new_hists = []
        new_ratings = []
        for row_number in tqdm(list(df_data.index)):
            row = df_data.loc[row_number]
            cur_id = int(row["Movie ID"])
            cur_indice = sorted_indice[cur_id - 1, :]
            cnt = 0
            hist_rating_dict = {
                hist: rating
                for hist, rating in zip(row["user_hist"], row["hist_rating"])
            }
            new_hist = []
            new_rating = []
            for index in cur_indice:  # index: int(raw_id)
                raw_index = str(index)
                if raw_index in hist_rating_dict:
                    cnt += 1
                    new_hist.append(raw_index)
                    new_rating.append(hist_rating_dict[raw_index])
                    if cnt == args.K:
                        break
            new_hists.append(new_hist)
            new_ratings.append(new_rating)
        df_data["user_hist"] = new_hists
        df_data["hist_rating"] = new_ratings
        df_data.to_parquet(
            os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz"),
            compression="gzip",
        )
    if not os.path.exists(
        os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
    ):
        df_data = pd.read_parquet(os.path.join(target_dir, f"{args.set}.parquet.gz"))
        df_data["origin_hist"] = df_data["user_hist"]
        df_data["origin_rating"] = df_data["hist_rating"]
        sorted_indice = np.load(
            os.path.join(embedding_dir, f"zero_shot_average_indice.npy")
        )  # from raw id , [3952, 3952]

        new_hists = []
        new_ratings = []
        for row_number in tqdm(list(df_data.index)):
            row = df_data.loc[row_number]
            cur_id = int(row["Movie ID"])
            cur_indice = sorted_indice[cur_id - 1, :]
            cnt = 0
            hist_rating_dict = {
                hist: rating
                for hist, rating in zip(row["user_hist"], row["hist_rating"])
            }
            hist_seq_dict = {hist: i for i, hist in enumerate(row["user_hist"])}
            new_hist = []
            new_rating = []
            for index in cur_indice:  # index: int(raw_id)
                raw_index = str(index)
                if raw_index in hist_rating_dict:
                    cnt += 1
                    new_hist.append(raw_index)
                    new_rating.append(hist_rating_dict[raw_index])
                    if cnt == args.K:
                        break
            zipped_list = sorted(
                zip(new_hist, new_rating), key=lambda x: hist_seq_dict[x[0]]
            )
            new_hist, new_rating = map(list, zip(*zipped_list))
            new_hists.append(new_hist)
            new_ratings.append(new_rating)
        df_data["user_hist"] = new_hists
        df_data["hist_rating"] = new_ratings
        df_data.to_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz"),
            compression="gzip",
        )
    assert args.set == "test"
    if args.data_type == "rethigh":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz")
        )
    elif args.data_type == "mine" or args.data_type == "GAARA":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
        )
    elif "rellaseq" in args.data_type or "rellasubseq" in args.data_type:
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
        )
    elif args.data_type == "retatt":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz")
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
    elif args.data_type == "retseq":
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
    elif args.data_type == "retfeat":
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_ret{args.K}.parquet.gz")
        )
        genre_encoder = LabelEncoder()

        movie_data = []
        movie_fields = ["Movie ID", "Movie title", "Movie genre"]
        for line in open(
            "../../data/ml-1m/raw_data/movies.dat", "r", encoding="ISO-8859-1"
        ).readlines():  # 扫描movie data
            ele = line.strip().split("::")
            movie_id = int(ele[0].strip())
            movie_title = ele[1].strip()
            movie_genre = ele[2].strip().split("|")[0]
            movie_data.append([movie_id, movie_title, movie_genre])

        df_movie = pd.DataFrame(movie_data, columns=movie_fields).set_index("Movie ID")

        df_movie["Movie genre"] = genre_encoder.fit_transform(df_movie["Movie genre"])

        df_data = pd.read_parquet(f"../../data/ml-1m/proc_data/{args.set}.parquet.gz")
        print(df_data.columns)
        # Load test data, which has history ID and history rating columns
        print(df_data.head(1))
        print(df_data.columns)
        genre_history = []
        genre_rating = []
        for idx, row in tqdm(df_data.iterrows()):
            target, hist = int(row["Movie ID"]), row["user_hist"][-args.K :]
            genre_hist = np.array(
                [df_movie.loc[int(hid)]["Movie genre"] for hid in hist]
            )
            genre_idx = np.argsort(genre_hist)
            genre_history.append(row["user_hist"][genre_idx])
            genre_rating.append(row["hist_rating"][genre_idx])

        df_data["user_hist"] = genre_history
        df_data["hist_rating"] = genre_rating
    elif "retsubseq" in args.data_type:
        df_data = pd.read_parquet(
            os.path.join(target_dir, f"{args.set}_rellaseq_{args.K}.parquet.gz")
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
    elif args.data_type == "retattseq":
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

    else:
        raise NotImplementedError
    df_data.to_parquet(
        os.path.join(target_dir, f"{args.set}_{args.data_type}_{args.K}.parquet.gz"),
        compression="gzip",
    )

DATA_DIR = f"../../data/{args.dataset}/proc_data/"

assert args.set in ["train", "test"]

fp = os.path.join(DATA_DIR, "data")
os.makedirs(fp, exist_ok=True)
fp = os.path.join(fp, args.set)
os.makedirs(fp, exist_ok=True)
file_name = "_".join(
    [args.set, f"{args.data_type}_{args.temp_type}_{args.K}", f"{args.chunk_interval}"]
)
fp = os.path.join(fp, file_name + ".json")
ori_data_fp = os.path.join(
    target_dir, f"{args.set}_{args.data_type}_{args.K}.parquet.gz"
)
df = pd.read_parquet(ori_data_fp)
start_idx, end_idx = int(args.chunk_interval.split(":")[0]), int(
    args.chunk_interval.split(":")[1]
)
print(f"getting sample from {start_idx} to {end_idx}")
df = df.iloc[start_idx:end_idx].reset_index()
df.to_parquet(
    os.path.join(
        target_dir,
        f"{args.set}_{args.data_type}_{args.K}_{args.chunk_interval}.parquet.gz",
    )
)


if args.dataset == "ml-1m":
    msg_iter = zero_shot_get_prompt(
        K=args.K,
        istrain=args.set,
        temp_type=temp_type,
        data_dir=target_dir,
        fp=f"{args.set}_{args.data_type}_{args.K}_{args.chunk_interval}.parquet.gz",
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
