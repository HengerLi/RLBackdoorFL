# RLBackdoorFL

Code for our paper for ICLR 2023 Workshop on Backdoor Attacks and Defenses in Machine Learning (BANDS): Learning to Backdoor Federated Learning
## setup environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```

## Code Structure
```
```DataProcess.py``` preprocesses data
```Aggr.py``` stores trainning-stage aggregation rules (i.e., defenses)
```Backdoor_attacks.py``` contains code for training autoencoder which is used in distribution learning.
```Networks.py``` contains network structures 
```Post_defenses.py``` contains post-trainning stage defenses
```Util.py``` includes all helper functions 
```

## Trainning attack policy
```
# Change the model dir to your own experiment
python3 train_cifar_krum_TD3_policy.py
python3 train_cifar_post_DDPG_policy.py
```

## Test
```
# Change the model dir to your own experiment
python3 cifar10_test.py
python3 cifar10_test1.py
python3 cifar10_test2.py
mnist_test.py
```

## Reference
```
The implementation is based on our Learning-to-Attack-Federated-Learning framework for untargeted model poisoning attack (NuerIPS'22)
[Paper link] https://openreview.net/pdf?id=4OHRr7gmhd4 
[Code link] https://github.com/HengerLi/Learning-to-Attack-Federated-Learning
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{li2023learning,
  title={Learning to Backdoor Federated Learning},
  author={Li, Henger and Wu, Chen and Zhu, Senchun and Zheng, Zizhan},
  journal={arXiv preprint arXiv:2303.03320},
  year={2023}
}
