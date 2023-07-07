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
    parser.add_argument('--surrogate_models', default=["inception_v4", "resnet18", "densenet161", "vgg16_bn"], type=str, nargs='+',
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
    parser.add_argument('--sgpu', default=3, type=int,
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
    same_num_list = [0 for i in range(len(surrogate_models)+1)]
    sum = 0
    sim12 = 0
    for step, (input, target) in enumerate(train_bar):
        input = input.to(device).float()
        target = target.to(device).long()
        at_target = torch.tensor(88).to(device).long()
        # at_target = at_target.repeat(input.size(0)).reshape_as(target)

        output = target_model(utils.norm_image(input))
        
        pre = torch.argmax(output,dim=-1).detach()
        correct += (pre==target).sum().cpu()
        correct_index = pre==target

        momentum = torch.zeros_like(input).detach().to(device)
        adv_images = input.clone().detach()

        sur_output = torch.zeros(output.size(0), len(surrogate_models), output.size(1)).type_as(output)


        for idx, (model) in enumerate(surrogate_models):
            output_tmp = model(utils.norm_image(input))
            sur_output[:,idx] = output_tmp

        _, sur_indx = torch.sort(sur_output)
        sur_max, _ = torch.max(sur_indx, dim=1)
        sur_min, _ = torch.min(sur_indx, dim=1)
        sur_o = sur_max - sur_min
        _, sur_target = torch.min(sur_o, dim=1)


        
        for i in range(attack_iter):
            adv_images = adv_images.detach()
            adv_images.requires_grad = True

            # white        
            output = target_model(utils.norm_image(adv_images))
            # loss = criterion(output, target)
            loss = criterion(output, sur_target)
            loss.backward()
            grad_true = adv_images.grad
            adv_images = adv_images.detach()
            adv_images.requires_grad = True

            
            # black
            grad_back = torch.zeros([input.shape[0],len(surrogate_models),input.shape[1],input.shape[2],input.shape[3]]).type_as(input)
            grad_and = torch.zeros_like(input)
            # # for model_index in range(len(surrogate_models)):


            for idx,(model) in enumerate(surrogate_models):
                model.zero_grad()
                output = model(utils.norm_image(adv_images))
                adv_pre = torch.argmax(output, dim=-1).detach()


                # loss = criterion(output, target)
                loss = criterion(output, sur_target)
                loss.backward()
                
                # u = u / torch.sqrt(torch.sum(u * u,dim=[1,2,3]))[:,None,None,None]
                # grad_back[:,idx] = adv_images.grad/ torch.max(torch.abs(adv_images.grad))
                grad_back[:,idx] = adv_images.grad/ torch.mean(torch.abs(adv_images.grad),dim=(1, 2, 3), keepdim=True)
                # grad_back[:,idx] = adv_images.grad

                adv_images = adv_images.detach()
                adv_images.requires_grad = True
                # # print(torch.nn.functional.normalize(adv_images.grad).shape)
                # adv_pre = torch.argmax(output, dim=-1).detach()
                indx_tmp = adv_pre!=target
                grad_back[:,idx][adv_pre!=target] = 0

                
                
            # grad_weight = mlp_grad(input)
            # print(grad_weight)
            # grad_weight = torch.full([input.shape[0],4],1/6).type_as(input)
            # grad_weight[:, 0] = 1/2
            # grad_weight = grad_weight[:,:,None,None,None]
            grad_weight = 1 / len(surrogate_models)
            grad_hat = torch.sum((grad_back*grad_weight),dim=1)

            # grad_back[:,0][grad_back[:,0].sign()!=grad_true.sign()] *= -1
            # index1 = (grad_back[:,0].sign() != grad_true.sign()) and (grad_hat.sign() == grad_true.sign())
            # grad_back[:,0][grad_temp!=10] = 0
            # print((grad_back[:,0]==0).sum()/16/3./224./224. )
            # print((grad_back[:,0].sign()!=grad_true.sign()).sum()/16/3./224./224. )
            # grad_hat[grad_true.sign()!=grad_hat.sign()] = 0
            # print((grad_hat!=0).sum()/16/3./224./224. )
            # print((grad_true.sign()!=grad_hat.sign()).sum()/16/3./224./224. )
            # grad_true[grad_true.sign()!=grad_hat.sign()] = -grad_true[grad_true.sign()!=grad_hat.sign()].sign()
            # grad_true = -grad_true
            # indx01 = (grad_back[:,0].sign() == grad_back[:,1].sign() )
            # indx02 = (grad_back[:,0].sign() == grad_back[:,2].sign() )
            # indx03 = (grad_back[:,0].sign() == grad_back[:,3].sign() )
            # indx12 = (grad_back[:,1].sign() == grad_back[:,2].sign() )
            # indx13 = (grad_back[:,1].sign() == grad_back[:,3].sign() )
            # indx23 = (grad_back[:,2].sign() == grad_back[:,3].sign() )
            # rate = (torch.sum(indx01 + indx02 + indx03 + indx12 + indx13 + indx23)) / indx01.size(0) / indx01.size(1) /indx01.size(2) / indx01.size(3)
            # # print("rate:"+str(rate))
            # grad_back[:,0][grad_back[:,1].sign() != grad_back[:,2].sign()] = 0
            # sim12 += torch.sum(indx12)  /  indx12.size(1) /indx12.size(2) / indx12.size(3)

            

            #ae 
            # grad_hat = mlp_grad(input)

            with torch.no_grad():
                for idx in range(len(surrogate_models)):
                    same_num = ((grad_true.sign() - grad_back[:,idx].sign()) == 0).sum().item()
                    same_num_list[idx] += same_num / (grad_hat.shape[1]*grad_hat.shape[2]*grad_hat.shape[3])
                    # print(same_num_list[idx])
                same_num = same_num = ((grad_true.sign() - grad_hat.sign()) == 0).sum().item()
                same_num_list[-1] += same_num / (grad_hat.shape[1]*grad_hat.shape[2]*grad_hat.shape[3])
                sum += grad_hat.shape[0]

                # grad = grad_back[:,0] # signal_0 model attack
                grad = grad_hat # ensemble attack
                # grad = torch.randn_like(adv_images).sign()  # random sign attack

                grad = grad / torch.mean(torch.abs(grad),dim=(1, 2, 3), keepdim=True)
                # grad = grad+ momentum * 1.0
                # momentum =grad

                adv_images = adv_images + per_iter_eps * grad.sign()
                delta = torch.clamp(adv_images - input, min=-eps, max=eps)
                adv_images = torch.clamp(input+ delta, min=0, max=1)

            # adv_output = target_model(utils.norm_image(adv_images))
            # adv_output = torch.argmax(adv_output,dim=-1).detach()
            # if adv_output != target:
            #     break

            
        
        # if input_size != val_size:
        #     resize_adv_images = F.interpolate(input=adv_images, size=val_size, mode='bicubic')
        #     output = target_model(resize_adv_images)
        # else:
        output = target_model(utils.norm_image(adv_images))
        
        # _, pre = torch.max(output.data, 1)
        pre = torch.argmax(output, -1)
        total += target.size(0)
        success_num += (pre[correct_index] != target[correct_index]).sum().cpu()

        train_bar.set_description(" step: [{}], acc: {:.4f} asr: {:.4f}".format(step, correct.item() / total *100,success_num.item() / correct.item() *100))

    for idx in range(len(surrogate_models)+1):
        print(f'model:{idx}, {same_num_list[idx]/sum}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Accuracy of model: {:.4f}".format(correct.item() / total *100))
    print("Accuracy of attack: {:.4f}".format(success_num.item() / correct.item() *100))
    # print("Accuracy of sim12: {:.4f}".format(sim12 / total *100))



if __name__ == '__main__':
    # torch.cuda.set_device('cuda:1')
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





