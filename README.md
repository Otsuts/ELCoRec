This this the code repository for paper: ELCoRec: Enhance Language Understanding with Co-Propagation of Numerical and Categorical Features for Recommendation. This paper has been accepted by CIKM 2024.

## 0. Environment
You can install nessesary packages by running:
```pip install -r requirements.txt```

## 1. Data preparation
First obtain the dataframes from raw data:
```
cd data_preprocess
python preprocess_ml_1m_timestamp.py
```

This will create ```train/valid/test.parquet.gz``` under directory ```data/ml-1m/proc_data/data/intermediate_data```

Before doing the following steps, make sure you have vicuna-13b model weight in directory ```models/vicuna```, otherwise, run the following commands to download it:
```
cd models
python download.py
```
This will download it from modelscope. And any other way to download the model is ok.

Next, build the indexed knowledge base by running:
```
cd data_preprocess
python emb_useritem_ml_1m.py
```
Item embeddings will be saved at ```data/ml-1m/PLM_data```

Then, calculate item similarity by runnning:
```
cd data_preprocess
python get_idx_sim.py ml-1m
```

Then, you can generate textual prompt. When generating train data, run:

```
cd data2json/generate_train_data
python data2json_ml_1m_sampled.py --data_type GAARA --temp_type RRAP --train_size 65536 --K 15 # For our data
python data2json_ml_1m_sampled.py --data_type GAARA_simple --temp_type RRAP --train_size 65536 --K 30 # For our data w/o RAP template
```

When generating test data, run:
```
cd data2json/generate_test_data
python data2json_ml1m_ret.py --data_type GAARA --temp_type RRAP --chunk_interval 0:-1 --K 15 # For our data
python data2json_ml1m_ret.py --data_type GRRRA --temp_type RRAP --chunk_interval 0:-1 --K 30 # For our data w/o RAP template
```

## 2. Pretarin expert GAT network
```
mkdir trained_models
mkdir trained_models/ml-1m
python train_gat.py --dataset ml-1m --lr <lr> --wd <wd>
```

After this, rename the saved model weight to ```GATGAARA.pth``` (we also have our trained_model in ```trained_models/ml-1m/GATGAARA.pth```)

## 3. Finetune
```
CUDA_VISIBLE_DEVICES=<your gpu id> python finetune_mymodel.py -- data_type <data_type>_<temp_type>_K --dataset ml-1m --K <K> --train_size <train_size>
```

## 4. Inference
```
CUDA_VISIBLE_DEVICES=<your gpu id> python inference_mymodel.py --data_type <data_type>_<temp_type>_K --dataset ml-1m --K <K> --chunk_interval <start_interval>_<end_interval>
```
## Citation
If you find this repo useful, please cite our paper.

@misc{chen2024elcorecenhancelanguageunderstanding,
      title={ELCoRec: Enhance Language Understanding with Co-Propagation of Numerical and Categorical Features for Recommendation}, 
      author={Jizheng Chen and Kounianhua Du and Jianghao Lin and Bo Chen and Ruiming Tang and Weinan Zhang},
      year={2024},
      eprint={2406.18825},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2406.18825}, 
}
