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
from utils.load import load_surrogate_model, load_target_model, load_transform
import timm


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
        self.softmax = torch.nn.Softmax()


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
    parser.add_argument('--name', default='fgsm', type=str,
                        help='attack method',
                        choices=['fgsm', 'mi_fgsm', 'ni_fgsm', 'vmi_fgsm'])
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--eps', default=16/255.0, type=float,
                        help='the maximum perturbation, linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--seed', default=3407, type=int)

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='dataset',
                        choices=['imagenet', 'cifar10', 'cifar100'])
    parser.add_argument('--root', default='/home/dataset/ImageNet/ILSVRC2012_img_val', type=str,
                        help='dataset path')
    parser.add_argument('--target_file', default='/home/liuhanpeng/at/data/val_rs.csv', type=str,
                        help='the figures attacked')
    parser.add_argument('--output_dir', default='./output_dir', type=str,
                        help='path where to save, empty for no saving')
    
    # Model parameters
    parser.add_argument('--model_1', default='inc_v3', type=str,
                        help='the white model')
    parser.add_argument('--model_2', default='inc_res_v2', type=str,
                        help='the white model')
    parser.add_argument('--model_3', default='inc_v4', type=str,
                        help='the white model')
    parser.add_argument('--model_4', default='inc_res_v2_ens', type=str,
                    help='the white model')
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
    eps = args.eps

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True
        torch.manual_seed(seed)
        torch.cuda.set_device(args.sgpu)
        print(torch.cuda.device_count())
        print('Using CUDA..')
    
    # if args.ngpu > 1:


    # load dataset
    print("load dataset!!!")
    transform, input_size = load_transform(args.model_1, args)
    dataset = Dataset(root=args.root, target_file=args.target_file, transform=transform)

    sampler_val = data.SequentialSampler(dataset)
    data_loader_val = data.DataLoader(dataset=dataset,sampler=sampler_val, batch_size=batch_size, shuffle=False, num_workers=8)
    print(data_loader_val)

    mseloss = torch.nn.MSELoss()
    if args.loss == 'ce':
        loss = nn.CrossEntropyLoss()
    
    
    # load model
    print("load model!!!")
    model_1 = load_surrogate_model(args.model_1)
    model_2 = load_surrogate_model(args.model_2)
    model_3 = load_surrogate_model(args.model_3)


    # load target_model
    print("load target_model!!!")
    target_model, val_size = load_target_model(args.target_model)
    
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
    model_3 = model_3.cuda()

    target_model = target_model.cuda()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    target_model.eval()

    print("Attack is start!!!")
    rgf = RGF(model=target_model, loss= loss, q=20, sigma=1e-4)
    mlp_g = MLP(3).cuda()
    mlp_g.train()
    opt = optim.SGD(mlp_g.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)

    total = 0
    correct =0
    train_bar = tqdm(data_loader_val)
    num = 0
    since = time.time()
    # for i, (input, true_target) in enumerate(train_bar):
    #     input = input.cuda().float()
    #     true_target = true_target.cuda().long()
    #     input.requires_grad = True
        
    #     g = torch.rand_like(input).cuda().float()
    #     for i in range(input.size(0)):
    #         g[i] = rgf.query(input[i].unsqueeze(0), true_target[i].unsqueeze(0))
        
    #     output_1 = model_1(input)
    #     cost = loss(output_1, true_target).cuda()
    #     cost.backward()
        
    #     grad_1 = input.grad
    #     model_1.zero_grad()


    #     output_2 = model_1(input)
    #     cost = loss(output_2, true_target)
    #     cost.backward()

    #     grad_2 = input.grad
    #     model_2.zero_grad()


    #     output_3 = model_3(input)
    #     cost = loss(output_3, true_target)
    #     cost.backward()

    #     grad_3 = input.grad
    #     model_3.zero_grad()

    #     # grad_s = torch.cat((grad_1, grad_2, grad_3), 1)
    #     # grad_s = grad_s.reshape(-1, 3*3*299*299)

    
    #     weight = mlp_g(input)
    #     g_hat = grad_1.reshape(-1, 3*299*299)*weight[:,1].reshape(-1,1) + grad_2.reshape(-1, 3*299*299) * weight[:,1:2].reshape(-1,1) + grad_3.reshape(-1, 3*299*299) * weight[:,2:3].reshape(-1,1)
    #     g_hat = g_hat.reshape(-1, 3*299*299)
    #     ct = mseloss(g_hat, g.reshape(-1, 3*299*299))


    #     opt.zero_grad()
    #     ct.backward()
    #     opt.step()
        


    #     train_bar.set_description(" epoch: [{}], asr: {:.4f}".format( i, ct.item()))
    #     # adv_images = input + 8/255 * images.grad.sign()
    #     # adv_images = torch.clamp(adv_images, 0, 1)
    
    # print('Saving..')
    # torch.save(mlp_g.state_dict(), os.path.join("./models/mlp_g.t7"))

    
    mlp_g.load_state_dict(torch.load("./models/mlp_g.t7"))
    for i, (input, true_target) in enumerate(train_bar):
        input = input.cuda()
        true_target = true_target.cuda()

        momentum = torch.zeros_like(input).detach().cuda()
        adv_images = input.clone().detach()
        

        for i in range(20):
            adv_images.requires_grad = True            
            output = target_model(adv_images)
            
            output_1 = model_1(adv_images)
            cost = loss(output_1, true_target).cuda()
            cost.backward()
            grad_1 = adv_images.grad
            model_1.zero_grad()


            output_2 = model_1(adv_images)
            cost = loss(output_2, true_target)
            cost.backward()
            grad_2 = adv_images.grad
            model_2.zero_grad()


            output_3 = model_3(adv_images)
            cost = loss(output_3, true_target)
            cost.backward()
            grad_3 = adv_images.grad
            model_3.zero_grad()
                
            # weight = mlp_g(adv_images)
            # grad =  grad_1.reshape(-1, 3*299*299)*weight[:,1].reshape(-1,1) + grad_2.reshape(-1, 3*299*299) * weight[:,1:2].reshape(-1,1) + grad_3.reshape(-1, 3*299*299) * weight[:,2:3].reshape(-1,1)
            grad =  grad_1.reshape(-1, 3*299*299)*1/3.0 + grad_2.reshape(-1, 3*299*299) * 1/3.0 + grad_3.reshape(-1, 3*299*299) * 1/3.0
            grad = grad.reshape(-1, 3, 299, 299)

            grad = grad / torch.mean(torch.abs(grad),
                                     dim=(1, 2, 3), keepdim=True)
            grad = grad+ momentum * 1.0
            momentum =grad

            adv_images = adv_images.detach() + 2/255 * grad.sign()
            delta = torch.clamp(adv_images - input, 
                                min=-8/255, max=8/255)
            adv_images = torch.clamp(input+ delta, min=0, max=1).detach()
        

        if input_size != val_size:
            resize_adv_images = F.interpolate(input=adv_images, size=val_size, mode='bicubic')
            output = target_model(resize_adv_images)
        else:
            output = target_model(adv_images)
        
        _, pre = torch.max(output.data, 1)

        total += true_target.size(0)

        correct += (pre != true_target).sum()

        train_bar.set_description(" epoch: [{}], asr: {:.4f}".format( i, correct.item() / total *100))



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Accuracy of attack: {:.4f}".format(correct.item()/total *100))



if __name__ == '__main__':
    # torch.cuda.set_device('cuda:1')
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





