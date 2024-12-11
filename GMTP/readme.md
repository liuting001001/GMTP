

![](./GMTP.png)

## Requirements

Our code is based on Python version 3.8 and PyTorch version 2.3.0+cu118, Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:
```
pip install -r requirements.txt
```

## Data

We conduct our experiments on two trajectory datasets and corresponding road networks, including **BJ** and **Porto**. We provide a link to download the processed dataset in Google Drive. [Click here](https://drive.google.com/file/d/14yTivaV41gst0_k4ufHBHSV205tcRNqb/view?usp=share_link) to download the zip files. If you want to use these datasets, you need to unzip them first. And, you need to create the raw_data/ directory manually. 

For example, if you unzip the **Porto** dataset, please make sure your directory structure is as follows:

- `GMTP/raw_data/porto_roadmap_edge_porto_True_1_merge/...`
- `GMTP/raw_data/porto/...`

Here `porto_roadmap_edge_porto_True_1_merge/` stores the road network data, and `porto/` stores the trajectory data.

for the Beijing trajectory dataset please refer to file [bj-data-introduction.md](./bj-data-introduction.md) for a more detailed data introduction. [Data Download](https://pan.baidu.com/s/1TbqhtImm_dWQZ1-9-1XsIQ?pwd=1231)

## Pre-Train
```shell
# Porto
python: run_model.py --model BERTContrastiveLM --dataset porto --config porto --gpu_id 0 --mlm_ratio 0.6 --contra_ratio 0.4 --split true --distribution geometric --avg_mask_len 2 --out_data_argument1 trim --out_data_argument2 shift
#618819
# BJ
python run_model.py --model BERTContrastiveLM --dataset bj --config bj --gpu_id 1 --mlm_ratio 0.6 --contra_ratio 0.4 --split true --distribution geometric --avg_mask_len 2 --out_data_argument1 trim --out_data_argument2 shift
```

The default data enhancement method is used here, i.e.,  *Trajectory perturbing* and *Trajectory masking*.

A field `exp_id` is generated to mark the experiment number during the experiment, and the pre-trained model will be stored at `libcity/cache/{exp_id}/model_cache/{exp_id}_{model_name}_{dataset}.pt`

 ## Fine-tune

We can fine-tune the model for downstream tasks. Note that you need to modify the `exp_id` field in the following command.

(1) Travel Time Prediction

```shell
# Porto
python run_model.py --model LinearETA --dataset porto --gpu_id 0 --config porto --pretrain_path libcity/cache/{exp_id}/model_cache/{exp_id}_BERTContrastiveLM_porto.pt
#10
# BJ
python run_model.py --model LinearETA --dataset bj --gpu_id 0 --config bj --pretrain_path libcity/cache/{exp_id}/model_cache/{exp_id}_BERTContrastiveLM_bj.pt
```

## Reference Code

The code references several open source repositories, to whom thanks are expressed here, including [START](https://github.com/aptx1231/start),[LibCity](https://github.com/LibCity/Bigscity-LibCity),[GATv2]{https://github.com/tech-srl/how_attentive_are_gats}, [pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT), [mvts_transformer](https://github.com/gzerveas/mvts_transformer), [ConSERT](https://github.com/yym6472/ConSERT),[CollabrateAttention](https://github.com/epfml/collaborative-attention).



