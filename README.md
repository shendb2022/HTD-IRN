# Hyperspectral Target Detection Based on Interpretable Representation Network (HTD-IRN), IEEE TGRS, 2023.

## files
### `Main.py`, `Model.py`, `Train_Test.py`,`ts_generation.py`,`utils.py`.

## Requirement
### `python 3.6.13`, `torch 1.10.1`,  NVIDIA Geforce RTX 2080Ti.

## Network
### HTD-IRN is a promising detector for hyperspectral imagery based on a deep subspace representation network with Uformer.

## Run
### run `Main.py`.

## Note
### 1. Three states of `train`, `test`, or `parameter_selection` can be chosen in `Main.py`.
### 2. We provide a well-trained model of San Diego I, and you can test it directly.
### 3. For a new dataset, the optimal values of m and eta1 should be chosen first. So you can change the state to `parameter_selection`  in `Main.py`.

## Cite
```
@ARTICLE{shen2023hyperspectral,
  author={Shen, Dunbin and Ma, Xiaorui and Kong, Wenfeng and Liu, Jianjun and Wang, Jie and Wang, Hongyu},
  journal={IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING}, 
  title={Hyperspectral Target Detection Based on Interpretable Representation Network}, 
  year={2023},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2023.3302950}}
```