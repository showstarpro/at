import os
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchvision.utils import save_image
import torch.nn as nn
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import os.path
import pandas as pd
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torch import autograd
import torch.optim as optim
import hashlib
import io
from attack.mi_fgsm import MI_FGSM
from attack.ni_fgsm import NI_FGSM
from attack.vmi_fgsm import VMI_FGSM
from query.rgf import RGF
from utils.dataset import Dataset
import attack
from torchvision import models
from tqdm import  tqdm
import torchvision.datasets as datasets
from pathlib import Path
from attack.fgsm import FGSM
from utils.load import load_model, load_target_model, load_transform
import timm
import utils
from ae import Based_AutoEncoder_More


class MLP(torch.nn.Module):
    def __init__(self, out_channels):
        super(MLP, self).__init__()

        self.cn1 = torch.nn.Conv2d(3,64,3)
        self.pl1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.cn2 = torch.nn.Conv2d(64,128,3)
        self.relu2 = torch.nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ln = nn.Linear(128,out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.cn1(x)
        x = self.relu1(x)
        x = self.pl1(x)
        x = self.cn2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.reshape(-1,128)
        x = self.ln(x)
        x = self.softmax(x)
        return x


def get_args_parser():
    parser = argparse.ArgumentParser('Attack in pytorch', add_help=False)
    parser.add_argument('--attack_name', default='fgsm', type=str,
                        help='attack method',
                        choices=['fgsm', 'mi_fgsm', 'ni_fgsm', 'vmi_fgsm'])
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--eps', default=8, type=float,
                        help='the maximum perturbation, linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--seed', default=3407, type=int)

    parser.add_argument('--save_dir', default='imagenet', type=str,help='filedir to save model')

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='dataset',
                        choices=['imagenet', 'cifar10', 'cifar100'])
    parser.add_argument('--dataset_path', default='/home/dataset/ImageNet/ILSVRC2012_img_val', type=str,
                        help='dataset path')
    parser.add_argument('--target_file', default='/home/liuhanpeng/at/data/val_rs.csv', type=str,
                        help='the figures attacked')
    parser.add_argument('--output_dir', default='./output_dir', type=str,
                        help='path where to save, empty for no saving')
    
    # Model parameters
    parser.add_argument('--surrogate_models', default=["inception_v4", "resnet18", "densenet161", "vgg16_bn", "vit_b_16", "vit_l_16"], type=str, nargs='+',
                        help='the surrogate_models list')
    parser.add_argument('--model_path', default=None, type=str, 
                        help='the path of white model')
    parser.add_argument('--target_model', default='vit_b_16', type=str,
                        help='the target model')
    parser.add_argument('--target_model_path', default=None, type=str,
                        help='the path of target model')
    parser.add_argument('--loss', default='ce', type=str,
                        help='loss for fgsm',
                        choices=['ce', 'cw'])
    
    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--ngpu', default=1, type=int,
                        help='number of gpu')
    parser.add_argument('--sgpu', default=0, type=int,
                        help='gpu index (start)')
    
    return parser


def main(args):
    seed = args.seed
    np.random.seed(seed)

    batch_size = args.batch_size
    # eps = args.eps/255

    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True
        torch.manual_seed(seed)
        torch.cuda.set_device(args.sgpu)
        print(torch.cuda.device_count())
        print(f'Using CUDA:{args.sgpu}')
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    # if args.ngpu > 1:


    # load dataset
    print("load dataset!!!")
    transform, input_size = load_transform(args=args)
    dataset = Dataset(root=args.dataset_path, target_file=args.target_file, transform=transform)

    sampler_val = data.SequentialSampler(dataset)
    data_loader_val = data.DataLoader(dataset=dataset,sampler=sampler_val, batch_size=batch_size, shuffle=False, num_workers=8)
    print(data_loader_val)

    mseloss = torch.nn.MSELoss()
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    # load model
    print("load models!!!")
    print(args.surrogate_models)
    surrogate_models = [load_model(model_name).to(device).eval() for model_name in args.surrogate_models]
    
    # load target_model
    print("load target_model!!!")
    val_size = 224
    print(args.target_model)
    target_model= load_model(args.target_model).to(device).eval()

    print("Attack is start!!!")
    rgf = RGF(model=target_model, loss= criterion, q=20, sigma=1e-4)
    # mlp_grad = Based_AutoEncoder_More().to(device).eval()
    mlp_grad = MLP(len(surrogate_models)).to(device).eval()
    # opt = optim.SGD(mlp_grad.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)

    eps=args.eps/255
    attack_iter = 10
    per_iter_eps = eps 

    total = 0
    correct =0
    success_num = 0
    train_bar = tqdm(data_loader_val)
    since = time.time()

    save_dir = args.save_dir
    save_path = os.path.join(save_dir,'mlp_grad_3.pkl')
    # mlp_grad.load_state_dict(torch.load(save_path))

    sim = torch.zeros(len(surrogate_models)).to(device).float()

    for step, (input, target) in enumerate(train_bar):
        input = input.to(device).float()
        target = target.to(device).long()
        input.requires_grad = True

        output = target_model(utils.norm_image(input))

        pre = torch.argmax(output,dim=-1).detach()
        correct += (pre==target).sum().cpu()
        

        loss = criterion(output, target)
        loss.backward()
        grad_target = input.grad.detach()
        grad_target = grad_target.reshape(input.size(0),-1)

        for idx, (model) in enumerate(surrogate_models):
            model.zero_grad()

            sur_input = input.clone().detach()
            sur_input.requires_grad = True

            sur_output = model(utils.norm_image(sur_input))
            sur_loss = criterion(sur_output, target)
            sur_loss.backward()
            sur_grad = sur_input.grad.detach()
            sur_grad = sur_grad.reshape(input.size(0), -1)
            sur_sim = torch.cosine_similarity(grad_target, sur_grad, dim=1)
            sur_sim = torch.sum(sur_sim)
            sim[idx] += sur_sim

        total += target.size(0)


        train_bar.set_description(" step: [{}], acc: {:.4f}".format(step, correct.item() / total *100))

    for idx in range(len(surrogate_models)):
        model_name = args.surrogate_models[idx]
        target_model_name = args.target_model
        print(model_name+"->"+target_model_name+":"+str(sim[idx]/total))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Accuracy of model: {:.4f}".format(correct.item() / total *100))



if __name__ == '__main__':
    # torch.cuda.set_device('cuda:1')
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





