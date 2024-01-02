# Hierarchical-Topic-aware-Contextualized-Transformers

Code and dataset for the paper "Hierarchical Topic-aware Contextualized Transformers". The paper is accepted by IEEE Transactions on Neural Networks and Learning Systems.

## Dependencies
The script has been tested running under Python 3.6.6, with the following packages installed (along with their dependencies):
- 'numpy == 1.18.0'
- 'tensorflow == 1.14.0'
- 'tqdm == 4.46.0'

In addition, CUDA 10.0 and cuDNN 7.4 are used.

## data processing
#### download dataset
run `getdata.sh` to downlaod the datasets in this paper. 
#### encode dataset
use the `src/encode.py` to encode the raw .txt files into the .npz form.


## download the base pre-trained gpt-2 model
We train Contextualized Transformers based on gpt-2-base (117M). Download the pre-trained model from gpt-2 official site or you can run the `download_model.py` we provided. 

## training
Now, you can train Contextualized Transformers! Just like this:

`python train_atten.py`

Change the dataset path in this script to reproduce the results of Table 1 in the paper.

Note that, we provide three scripts (`trian_atten.py`, `train_embed.py`, `train_virtual.py`) corresponding to three modules in the paper (`topic attention`, `token embedding`, and `segment embedding`)
|
## Customize your own dataset
Topic-aware Contextualized Transformers can be trained on any text datasets. 

## Cite Our Paper
Lu R, Chen B, Guo D, et al. Hierarchical Topic-Aware Contextualized Transformers[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

