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
        # self.weights_init_kaiming()


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

    def weights_init_kaiming(self):
        for m in self.modules():
            classname = m.__class__.__name__
            # print(classname)
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in',nonlinearity='relu')
            elif classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out',nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm1d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


    
    

def get_args_parser():
    parser = argparse.ArgumentParser('Attack in pytorch', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
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
    parser.add_argument('--surrogate_models', default='inc_v3', type=str, nargs='+',
                        help='the surrogate_models list')
    parser.add_argument('--model_path', default=None, type=str, 
                        help='the path of white model')
    parser.add_argument('--target_model', default='resnet101', type=str,
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
    parser.add_argument('--sgpu', default=3, type=int,
                        help='gpu index (start)')
    
    return parser



def main(args):
    seed = args.seed
    np.random.seed(seed)

    batch_size = args.batch_size
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True
        torch.manual_seed(seed)
        torch.cuda.set_device(args.sgpu)
        print(torch.cuda.device_count())
        print('Using CUDA..')
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
    print("load model!!!")
    # print(args.surrogate_models)
    # surrogate_models = [load_model(model_name).to(device).eval() for model_name in args.surrogate_models]
    
    # load target_model
    print("load target_model!!!")
    val_size = 224
    target_model= load_model(args.target_model).to(device).eval()

    print("Attack is start!!!")
    rgf = RGF(model=target_model, loss=criterion, q=20, sigma=1e-4)
    mlp_grad = Based_AutoEncoder_More().to(device).train()
    # mlp_grad = MLP(len(surrogate_models)).to(device).train()
    opt = optim.SGD(mlp_grad.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    save_dir = args.save_dir
    eps = 16/255
    attack_iter = 10
    iter_eps = eps/attack_iter


    # train_bar = tqdm(data_loader_val)
    max_epoch = 10
    for epoch in range(max_epoch):
        train_bar = tqdm(data_loader_val)
        loss_sum = 0
        acc_list = []
        sum = 0
        for i, (input, target) in enumerate(train_bar):
            input = input.to(device).float()
            target = target.to(device).long()
            input.requires_grad = True
            adv_images = input.clone()
            # grad_query = torch.zeros_like(input)
            # for i in range(input.size(0)):
            #     grad_query[i] = rgf.query(input[i:i+1], target[i:i+1])
            # grad_query = rgf.query(input,target)
            for attack_i in range(10):
                adv_images = adv_images.detach()
                adv_images.requires_grad = True
                sum+=adv_images.shape[0]
                output = target_model(utils.norm_image(adv_images))
                loss = criterion(output, target)
                loss.backward()
                grad_query = adv_images.grad.clone().detach()
                with torch.no_grad():
                    grad = grad_query
                    grad = grad / torch.mean(torch.abs(grad),dim=(1, 2, 3), keepdim=True)

                    adv_images = adv_images + iter_eps * grad.sign()
                    delta = torch.clamp(adv_images - input, min=-eps, max=eps)
                    adv_images = torch.clamp(input+ delta, min=0, max=1)

                # # adv_images.grad.zero_()
                adv_images.requires_grad = True
                # grad_back = torch.zeros([input.shape[0],len(surrogate_models),input.shape[1],input.shape[2],input.shape[3]]).type_as(input)
                # # for model_index in range(len(surrogate_models)):
                # for idx,(model) in enumerate(surrogate_models):
                #     # input = input.detach()
                #     model.zero_grad()
                #     output = model(utils.norm_image(adv_images))
                #     loss = criterion(output, target)
                #     loss.backward()
                #     grad_back[:,idx] = adv_images.grad
                
                
                # # grad_s = torch.cat((grad_1, grad_2, grad_3), 1)
                # # grad_s = grad_s.reshape(-1, 3*3*299*299)
                # grad_weight = mlp_grad(adv_images)
                # grad_weight = grad_weight[:,:,None,None,None]
                # grad_hat = torch.sum((grad_back*grad_weight),dim=1)
                # # g_hat = g_hat.reshape(-1, 3*299*299)
                # loss_grad = mseloss(grad_hat,grad_query)

                grad_learn = mlp_grad(adv_images)
                loss_grad = mseloss(grad_learn,grad_query.sign())

                opt.zero_grad()
                loss_grad.backward()
                opt.step()

                sign_learn = grad_query.sign()
                sign_query = grad_learn.sign()
                # same_num = (sign_learn == sign_query).sum(dim=(1,2,3)).cpu().numpy()
                same_num = ((sign_learn - sign_query) == 0).sum(dim=(1,2,3)).cpu().numpy()
                loss_sum += loss_grad.item()
                acc_list.append(same_num)
            acc_np = np.concatenate(acc_list,axis=0).mean()
            acc_np = acc_np / (grad_query.shape[1]*grad_query.shape[2]*grad_query.shape[3])
            
            train_bar.set_description("epoch: {} step: [{}], loss: {:.8f} acc: {:.8f}".format(epoch, i, loss_sum/sum,acc_np))
            # adv_images = input + 8/255 * images.grad.sign()
            # adv_images = torch.clamp(adv_images, 0, 1)
        
    print('Saving..')
    save_path = os.path.join(save_dir,'ae_grad.pkl')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(mlp_grad.state_dict(), save_path)
    print(f'model save in {save_path}')



if __name__ == '__main__':
    # torch.cuda.set_device('cuda:1')
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





