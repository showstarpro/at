# Some adversarial-attacks in Pytorch

### Requirements 

- python==3.9.16
- Pytorch==1.12.1
- timm
- torchvision==0.13.0


### dataset

We randomly pick 1,000 clean images pertaining to the 1,000 categories from the ILSVRC 2012 validation set, which are almost correctly classified by all the testing models.

dataset_root = ./val_rs.csv


### attack_method

FGSM、PGD、MI-FGSM、NI-FGSM、VMI-FGSM are placed in ./attack

### use the attack method

python transform_mi_fgsm.py --model inc_v3 --target_model inc_v4

