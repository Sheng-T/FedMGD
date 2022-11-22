# Modeling Global Distribution for Federated Learning with Label Distribution Skew

## Abstract
Federated learning achieves joint training of deep models by connecting decentralized data sources, which can significantly mitigate the risk of privacy leakage. However, in a more general case, the distributions of labels among clients are different, called "label distribution skew". Directly applying conventional federated learning without consideration of label distribution skew issue significantly hurts the performance of the global model. To this end, we propose a novel federated learning method, named FedMGD, to alleviate the performance degradation caused by the label distribution skew issue. It introduces a global Generative Adversarial Network to model the global data distribution without access to local datasets, so the global model can be trained using the global information of data distribution without privacy leakage. The experimental results demonstrate that our proposed method significantly outperforms the state-of-the-art on several public benchmarks.

<img src="https://raw.githubusercontent.com/LuftmenschDevil/FedMGD/master/img/Federated%20Enhancement%20Stage.jpg" width=256 height=256 alt="Generative Adversarial Stage" />
<img src="https://raw.githubusercontent.com/LuftmenschDevil/FedMGD/master/img/Generative%20Adversarial%20Stage.jpg" width=256 height=256 alt="Federated Enhancement Stage" />

## Requirements
- torch
- torchvision
- dominate
- visdom
- h5py
- numpy
- matplotlib

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

|   Dataset    |  α   | Epoch |                            Model                             |                          Generator                           |
| :----------: | :--: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| FashionMNIST | 0.01 |  100  | [download](https://pan.baidu.com/s/1qFzJhnRyghaC8EhVA4Bfuw?pwd=za49) | [download](https://pan.baidu.com/s/1WEps_UjWjBbSp4pqPVptuA?pwd=ygs0) |
| FashionMNIST | 0.05 |  100  | [download](https://pan.baidu.com/s/1St02ocZZ3HQqUml57uFRGA?pwd=si4k) | [download](https://pan.baidu.com/s/1zcyBPSI6JkBuw1C4PLPo2w?pwd=aqck) |
| FashionMNIST | 0.1  |  100  | [download](https://pan.baidu.com/s/15jkOFWZBdykVC6Hbwu1EzA?pwd=nxz7) | [download](https://pan.baidu.com/s/1bv_JCimkIafcFIrJK9a-7w?pwd=c39n) |


## Acknowledgments
We refer to the structure of [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to build the code.