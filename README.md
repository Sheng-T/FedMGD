# Modeling Global Distribution for Federated Learning with Label Distribution Skew

## Abstract
Federated learning achieves joint training of deep models by connecting decentralized data sources, which can significantly mitigate the risk of privacy leakage. However, in a more general case, the distributions of labels among clients are different, called "label distribution skew". Directly applying conventional federated learning without consideration of label distribution skew issue significantly hurts the performance of the global model. To this end, we propose a novel federated learning method, named FedMGD, to alleviate the performance degradation caused by the label distribution skew issue. It introduces a global Generative Adversarial Network to model the global data distribution without access to local datasets, so the global model can be trained using the global information of data distribution without privacy leakage. The experimental results demonstrate that our proposed method significantly outperforms the state-of-the-art on several public benchmarks.


![FedMGD](https://raw.githubusercontent.com/LuftmenschDevil/FedMGD/master/img/FedMGD.png)


## Requirements
```conda
conda create -n fedmgd python=3.7
pip install -r requirements.txt
```

## Generate Datasets
-   Download dataset (EMNIST / FashionMNIST / SVHN / CIFAR10), and divide data to the client. 

For example:
> cd ./data/EMNIST
>
> python generate_niid_dirichlet.py
- Collect distillation data.
> pyhton upload_0.05_data.py

## Get Started
### Example: the EMNIST dataset

#### 1.Comparison with State-of-the-art Methods.
- FedMGD
```
  python train_fedmgd.py    --dataroot your_data_root \
                            --name fedmgd \ 
                            --gpu_ids 0,1 \
                            --checkpoints_dir ./result \
                            --model fedmgd  \
                            --input_nc 1 \
                            --n_class 26 \
                            --n_client 5 \
                            --rounds 100 \
                            --num_epochs 10 \
                            --lr 0.0002 \
                            --lr_G 0.0002 \
                            --lr_D 0.0002 
```
-   Other Methods (In the case of FedAvg, please change the parameter "model")
```
  python train_federated.py --dataroot your_data_root \
                            --name fedavg \ 
                            --gpu_ids 0,1 \
                            --checkpoints_dir ./result \
                            --model fedavg  \
                            --input_nc 1 \
                            --n_class 26 \
                            --n_client 5 \
                            --rounds 100 \
                            --num_epochs 10 \
                            --lr 0.0002 \
```
#### 2.Ablation Experiment.
- (1) Realistic Score in FedMGD. 

    -  FedDF(unlabeled real data)
        ```
       python train_federated.py --model feddf --dataroot your_data_root
       ```
       
    -  FedDF(labeled real data)
        ```
       python train_federated.py --model feddf_with_label --dataroot your_data_root
       ```

    -  FedMGD+FedDF(w/o real data)   
        ```
       python train_fedmgd.py --model fedmgd_feddf --dataroot your_data_root
       ```

    -  F2U+FedDF(unlabeled w/o real data)
        ```
       python train_gan.py --model f2u --dataroot your_data_root
       python train_federated.py --model feddf --dataroot your_data_root --G_path f2u_model_path
       ```
    
-   (2) Compare with other distributed GANs.
       ```
       python train_gan.py --model mdgan --dataroot your_data_root
    
       python train_gan.py --model fedgan --dataroot your_data_root
    
       python train_fedmgd.py --model fedmgd --dataroot your_data_root
       ```
## Models
| Dataset |  α   |  Epoch  |                            Model                             |                          Generator                           |         Acc(%)         |
| :-----: | :--: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------: |
| EMNIST  | 0.01 | 400+100 | [download](https://pan.baidu.com/s/1jZtpV8FCiVF-LUxJxSuY0g?pwd=5qkz) | [download](https://pan.baidu.com/s/1zs3qG3awVMXyjF9Do5os6w?pwd=lbql) | **89.00±0.93 (↑2.44)** |
| EMNIST  | 0.05 | 400+100 | [download](https://pan.baidu.com/s/1kU_4kVhPPstzC_mJEHs2aQ?pwd=drlb) | [download](https://pan.baidu.com/s/1Q5ShOo_sL4d6d0u89plr4A?pwd=yvph) | **91.15±0.35 (↑1.82)** |
| EMNIST  | 0.1  | 400+100 | [download](https://pan.baidu.com/s/1Fcscuflov2T223cjvI4GWQ?pwd=u4go) | [download](https://pan.baidu.com/s/1FKCZsnO_hloC_jUgGkFOhw?pwd=10mx) |   91.52±0.17 (↓0.13)   |



| Dataset |  α   |  Epoch  |                            Model                             |                          Generator                           |         Acc(%)         |
| :-----: | :--: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------: |
|  SVHN   | 0.01 | 400+100 | [download](https://pan.baidu.com/s/1y70X-1-UIEice9_GWnNdBg?pwd=h80g) | [download](https://pan.baidu.com/s/14UoJiWeCDUkFxiY8pvm6og?pwd=q47i) | **84.14±0.91 (↑1.09)** |
|  SVHN   | 0.05 | 400+100 | [download](https://pan.baidu.com/s/1Gy3I8necFrjfl4TeMLKCXg?pwd=spo9) | [download](https://pan.baidu.com/s/1J0_dhEkX6G_-mNkeznHWVQ?pwd=mfsz) | **88.62±0.39 (↑0.96)** |
|  SVHN   | 0.1  | 400+100 | [download](https://pan.baidu.com/s/1lBoDys-DA5M28xkRkcSHzA?pwd=26os) | [download](https://pan.baidu.com/s/1S7_WtYUhK29nL_boG_8C7w?pwd=69ms) | **90.47±0.37 (↑0.64)** |


| Dataset |  α   |  Epoch  |                            Model                             |                          Generator                           |         Acc(%)         |
| :-----: | :--: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------: |
| Cifar10 | 0.01 | 400+100 | [download](https://pan.baidu.com/s/1JuNDdv8721gw2pAxgPjFvg?pwd=2381) | [download](https://pan.baidu.com/s/1UXmCm8NStEpYach_14ezLA?pwd=uncu) | **62.61±1.54 (↑8.15)** |
| Cifar10 | 0.05 | 400+100 | [download](https://pan.baidu.com/s/1uM9PCbSC9XTs-2hLfgBPUA?pwd=llml) | [download](https://pan.baidu.com/s/1bHVuOc5yAsafIJ9_qLP18A?pwd=5har) | **66.52±0.49 (↑2.24)** |
| Cifar10 | 0.1  | 400+100 | [download](https://pan.baidu.com/s/1o3gLjwRnZbCDneeCO531rA?pwd=2hu9) | [download](https://pan.baidu.com/s/1U4yeTv731Tn6gzkVP67BhQ?pwd=sxc1) | **69.31±0.33 (↑1.94)** |


|   Dataset    |  α   |  Epoch  |                            Model                             |                          Generator                           |         Acc(%)          |
| :----------: | :--: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------: |
| FashionMNIST | 0.01 | 400+100 | [download](https://pan.baidu.com/s/1qFzJhnRyghaC8EhVA4Bfuw?pwd=za49) | [download](https://pan.baidu.com/s/1WEps_UjWjBbSp4pqPVptuA?pwd=ygs0) | **84.04±0.58 (↑13.02)** |
| FashionMNIST | 0.05 | 400+100 | [download](https://pan.baidu.com/s/1St02ocZZ3HQqUml57uFRGA?pwd=si4k) | [download](https://pan.baidu.com/s/1zcyBPSI6JkBuw1C4PLPo2w?pwd=aqck) | **87.57±0.40 (↑3.90)**  |
| FashionMNIST | 0.1  | 400+100 | [download](https://pan.baidu.com/s/15jkOFWZBdykVC6Hbwu1EzA?pwd=nxz7) | [download](https://pan.baidu.com/s/1bv_JCimkIafcFIrJK9a-7w?pwd=c39n) | **89.29±1.33 (↑1.01)**  |

## Citing this work

```
@article{sheng2022modeling,
  title={Modeling Global Distribution for Federated Learning with Label Distribution Skew},
  author={Sheng, Tao and Shen, Chengchao and Liu, Yuan and Ou, Yeyu and Qu, Zhe and Yixiong, Liang and Wang, Jianxin},
  journal={arXiv preprint arXiv:2212.08883},
  year={2022}
}
```


## Acknowledgments
We refer to the structure of [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to build the code.