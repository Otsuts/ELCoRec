import pandas as pd
import numpy as np
import json
import os
import re
import random
import copy
from transformers import set_seed
import hashlib
import json
import pickle as pkl
import h5py
import collections
from tqdm import tqdm

set_seed(42)

dataset = "ml-1m"
root = f"../data/{dataset}"
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")

age_dict = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "above 56",
}

job_dict = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

# User data

user_data = []
user_fields = ["User ID", "Gender", "Age", "Job", "Zipcode"]
for line in open(os.path.join(source_dir, "users.dat"), "r").readlines():
    ele = line.strip().split("::")
    user_id, gender, age, job, zipcode = [x.strip() for x in ele]
    # assert gender in ["M", "F"], ele
    gender = "male" if gender == "M" else "female"
    age = age_dict[int(age)]
    job = job_dict[int(job)]
    user_data.append([user_id, gender, age, job, zipcode])

df_user = pd.DataFrame(user_data, columns=user_fields)
df_user.to_parquet(
    "../data/ml-1m/proc_data/data/intermediate_data/users.parquet.gz",
    compression="gzip",
)
print(f"Total number of users: {len(df_user)}")


# Movie data

movie_data = []
movie_fields = ["Movie ID", "Movie title", "Movie genre"]
for line in open(
    os.path.join(source_dir, "movies.dat"), "r", encoding="ISO-8859-1"
).readlines():
    ele = line.strip().split("::")
    movie_id = ele[0].strip()
    movie_title = ele[1].strip()
    movie_genre = ele[2].strip().split("|")[0]
    movie_data.append([movie_id, movie_title, movie_genre])

df_movie = pd.DataFrame(movie_data, columns=movie_fields)
# df_movie.to_parquet(
#     "../data/ml-1m/proc_data/data/intermediate_data/movies.parquet.gz",
#     compression="gzip",
# )
print(f"Total number of movies: {len(df_movie)}")


# Rating data

rating_data = []
rating_fields = ["User ID", "Movie ID", "rating", "timestamp", "labels"]
user_list, movie_list = list(df_user["User ID"]), list(df_movie["Movie ID"])
for line in open(os.path.join(source_dir, "ratings.dat"), "r").readlines():
    ele = [x.strip() for x in line.strip().split("::")]
    user, movie, rating, timestamp = ele[0], ele[1], int(ele[2]), int(ele[3])
    label = 1 if rating > 3 else 0
    if user in user_list and movie in movie_list:
        rating_data.append([user, movie, rating, timestamp, label])

df_ratings = pd.DataFrame(rating_data, columns=rating_fields)
print(f"Total number of ratings: {len(df_ratings)}")


# Merge df_user/df_movie/df_rating into df_data

df_data = pd.merge(df_ratings, df_user, on=["User ID"], how="inner")
df_data = pd.merge(df_data, df_movie, on=["Movie ID"], how="inner")

df_data.sort_values(
    by=["timestamp", "User ID", "Movie ID"], inplace=True, kind="stable"
)

field_names = [
    "timestamp",
    "User ID",
    "Gender",
    "Age",
    "Job",
    "Zipcode",
    "Movie ID",
    "Movie title",
    "Movie genre",
    "rating",
    "labels",
]

df_data = df_data[field_names].reset_index(drop=True)


df_data.head()

# Encode the feature dict for CTR data


def add_to_dict(dict, feature):
    if feature not in dict:
        dict[feature] = len(dict)


field_names = [
    "User ID",
    "Gender",
    "Age",
    "Job",
    "Zipcode",
    "Movie ID",
    "Movie title",
    "Movie genre",
]
feature_dict = {field: {} for field in field_names}


for idx, row in tqdm(df_data.iterrows()):
    for field in field_names:
        add_to_dict(feature_dict[field], row[field])

feature_count = [len(feature_dict[field]) for field in field_names]

feature_offset = [0]
for c in feature_count[:-1]:
    feature_offset.append(feature_offset[-1] + c)

for field in field_names:
    print(field, len(feature_dict[field]))


print("---------------------------------------------------------------")
for f, fc, fo in zip(field_names, feature_count, feature_offset):
    print(f, fc, fo)
print("---------------------------------------------------------------")


# Collect user history

user_history_dict = {
    "ID": {k: [] for k in set(df_data["User ID"])},
    "rating": {k: [] for k in set(df_data["User ID"])},
    "hist_timestamp": {k: [] for k in set(df_data["User ID"])},
}
history_column = {
    "ID": [],
    "rating": [],
    "hist_timestamp": [],
}
movie_id_to_title = {}

for idx, row in tqdm(df_data.iterrows()):
    user_id, movie_id, rating, title, timestamp = (
        row["User ID"],
        row["Movie ID"],
        row["rating"],
        row["Movie title"],
        row["timestamp"],
    )
    history_column["ID"].append(user_history_dict["ID"][user_id].copy())
    history_column["rating"].append(user_history_dict["rating"][user_id].copy())
    history_column["hist_timestamp"].append(
        user_history_dict["hist_timestamp"][user_id].copy()
    )
    user_history_dict["ID"][user_id].append(movie_id)
    user_history_dict["rating"][user_id].append(rating)
    user_history_dict["hist_timestamp"][user_id].append(timestamp)
    if movie_id not in movie_id_to_title:
        movie_id_to_title[movie_id] = title

# json.dump(movie_id_to_title, open(os.path.join(target_dir, "id_to_title.json"), "w"))

# Drop data sample with history length that is less than 5.

df_data["user_hist"] = history_column["ID"]
df_data["hist_rating"] = history_column["rating"]
df_data["hist_timestamp"] = history_column["hist_timestamp"]

df_data = df_data[df_data["user_hist"].apply(lambda x: len(x)) >= 5].reset_index(
    drop=True
)

history_column["ID"] = [x for x in history_column["ID"] if len(x) >= 5]
history_column["rating"] = [x for x in history_column["rating"] if len(x) >= 5]
history_column["hist_timestamp"] = [
    x for x in history_column["hist_timestamp"] if len(x) >= 5
]
history_column["hist length"] = [len(x) for x in history_column["rating"]]

for idx, row in tqdm(df_data.iterrows()):
    assert row["user_hist"] == history_column["ID"][idx]
    assert row["hist_rating"] == history_column["rating"][idx]
    assert row["hist_timestamp"] == history_column["hist_timestamp"][idx]
    assert len(row["hist_rating"]) == history_column["hist length"][idx]


print(df_data.head(10))
print(f"Number of data sampels: {len(df_data)}")


# Split & save user history sequence

train_num = int(0.8 * len(df_data))
valid_num = int(0.1 * len(df_data))
test_num = len(df_data) - train_num - valid_num

# user_seq = {
#     "history ID": {
#         "train": history_column["ID"][:train_num],
#         "valid": history_column["ID"][train_num : train_num + valid_num],
#         "test": history_column["ID"][train_num + valid_num :],
#     },
#     "history rating": {
#         "train": history_column["rating"][:train_num],
#         "valid": history_column["rating"][train_num : train_num + valid_num],
#         "test": history_column["rating"][train_num + valid_num :],
#     },
#     "history length": {
#         "train": history_column["hist length"][:train_num],
#         "valid": history_column["hist length"][train_num : train_num + valid_num],
#         "test": history_column["hist length"][train_num + valid_num :],
#     },
# }

# Save train/valid/test in parquet format

df_train = df_data[:train_num].reset_index(drop=True)
df_valid = df_data[train_num : train_num + valid_num].reset_index(drop=True)
df_test = df_data[train_num + valid_num :].reset_index(drop=True)


assert len(df_train) == train_num
assert len(df_valid) == valid_num
assert len(df_test) == test_num

print(f"Train num: {len(df_train)}")
print(f"Valid num: {len(df_valid)}")
print(f"Test num: {len(df_test)}")

df_train.to_parquet(os.path.join(target_dir, "train_ts.parquet.gz"), compression="gzip")
df_valid.to_parquet(os.path.join(target_dir, "valid_ts.parquet.gz"), compression="gzip")
df_test.to_parquet(os.path.join(target_dir, "test_ts.parquet.gz"), compression="gzip")
assert 0

print(df_test.head(10))
for i in range(10):
    print(df_test.loc[i]["user_hist"])

# Save the meta data for CTR

meta_data = {
    "field_names": field_names,
    "feature_count": feature_count,
    "feature_dict": feature_dict,
    "feature_offset": feature_offset,
    "movie_id_to_title": movie_id_to_title,
    "num_ratings": 5,
}

md5_hash = hashlib.md5(
    json.dumps(meta_data, sort_keys=True).encode("utf-8")
).hexdigest()
print("meta_data", md5_hash)

json.dump(
    meta_data, open(os.path.join(target_dir, "ctr-meta.json"), "w"), ensure_ascii=False
)
