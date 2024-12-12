

![](./GMTP.png)

## Requirements

Our code is based on Python version 3.8 and PyTorch version 2.3.0+cu118, Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:
```
pip install -r requirements.txt
```

## Data

We conduct our experiments on two trajectory datasets and corresponding road networks, including **BJ** and **Porto**. We provide a link to download the processed dataset in Google Drive. [Click here](https://github.com/aptx1231/start) to download the zip files. If you want to use these datasets, you need to unzip them first. And, you need to create the raw_data/ directory manually. 


## Reference Code

The code references several open source repositories, to whom thanks are expressed here, including [START](https://github.com/aptx1231/start),[LibCity](https://github.com/LibCity/Bigscity-LibCity),[GATv2]{https://github.com/tech-srl/how_attentive_are_gats}, [pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT), [mvts_transformer](https://github.com/gzerveas/mvts_transformer), [ConSERT](https://github.com/yym6472/ConSERT),[CollabrateAttention](https://github.com/epfml/collaborative-attention).



