# Modeling Global Distribution for Federated Learning with Label Shift

## Abstract
Federated learning achieves joint training of deep models by connecting decentralized data sources, which can significantly mitigate the risk of privacy leakage. However, in a more general case, the distributions of labels among clients is different, called "label shift". Directly applying conventional federated learning without consideration of label shift issue significantly hurts the performance of global model. To this end, we propose a novel federated learning method, named FedMGD, to alleviate the performance degradation caused by label shift issue. It introduces a global Generative Adversarial Network to model the global data distribution without access to local data of clients, so the global model can be trained using the global information of data distribution without privacy leakage. The experimental results demonstrate that our proposed method significantly outperforms the state-of-the-art on several public benchmarks.

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

#### 1.Experiments to verify FedMGD performance in label shift.
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
#### 2.Experiments to evaluate the quality of the generated images.
- (1) Replace the Collection of Data. 
To FedDF(unlabeled real data)/ FedDF(labeled real data)

    `python train_federated.py --model feddf --dataroot your_data_root`  

    `python train_federated.py --model feddf_with_data --dataroot your_data_root`
  
    `python train_fedmgd.py --model fedmgd_feddf --dataroot your_data_root`

-   (2) Compare with other distributed GANs.

    `python train_gan.py --model mdgan --dataroot your_data_root`  
    
    `python train_gan.py --model fedgan --dataroot your_data_root` 
    
    `python train_fedmgd.py --model fedmgd --dataroot your_data_root` 