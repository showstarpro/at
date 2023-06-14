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
import hashlib
import io
from utils.dataset import Dataset
import attack
from torchvision import models
from tqdm import  tqdm
import torchvision.datasets as datasets
from pathlib import Path
from attack.ni_fgsm import NI_FGSM
import timm

def get_args_parser():
    parser = argparse.ArgumentParser('NI_FGSM Attack in pytorch', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--eps', default=16/255.0, type=float,
                        help='the maximum perturbation, linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--alpha', default=1.6, type=float,
                        help='the alpha')
    parser.add_argument('--step', default=10, type=int,
                        help='the step')
    parser.add_argument('--decay', default=1.0, type=float,
                        help='the decay')

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
    parser.add_argument('--model', default='resnet101', type=str,
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
    if args.dataset == 'imagenet':
        if args.model == 'resnet101':
            transform =  transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_size = 224
        elif args.model in ['inc_v3', 'inc_v4','inc_res_v2', 'inc_res_v2_ens', 'inc_v3_adv']:
            transform =  transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_size = 229

            
    dataset = Dataset(root=args.root, target_file=args.target_file, transform=transform)

    sampler_val = data.SequentialSampler(dataset)
    data_loader_val = data.DataLoader(dataset=dataset,sampler=sampler_val, batch_size=batch_size, shuffle=False, num_workers=8)
    print(data_loader_val)

    if args.loss == 'ce':
        loss = nn.CrossEntropyLoss()
    
    # load model
    print("load model!!!")
    if args.model == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif args.model == 'inc_v4':
        model = timm.create_model('inception_v4', pretrained=True)
    elif args.model == 'inc_res_v2':
        model = timm.create_model('inception_resnet_v2', pretrained=True)
    elif args.model == 'inc_res_v2_ens':
        model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=True)
    elif args.model == 'inc_v3':
        model = models.inception_v3(pretrained=True)
    elif args.model == 'inc_v3_adv':
        model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    

    # load target_model
    print("load target_model!!!")
    val_size = 299
    if args.target_model == 'inc_v3':
        target_model = models.inception_v3(pretrained=True)     
    elif args.target_model == 'inc_v4':
        target_model = timm.create_model('inception_v4', pretrained=True)
    elif args.target_model == 'inc_res_v2':
        target_model = timm.create_model('inception_resnet_v2', pretrained=True)
    elif args.target_model == 'inc_res_v2_ens':
        target_model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=True)
    elif args.target_model == 'inc_v3_adv':
        target_model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    elif args.target_model == 'resnet101':
        target_model = models.resnet101(pretrained=True)
        val_size = 224
    

    
    model = model.cuda()
    target_model = target_model.cuda()
    model.eval()
    target_model.eval()

    print("Attack is start!!!")
    attack = NI_FGSM(model, loss, eps, alpha=args.alpha, steps=args.step, decay=args.decay)


    total = 0
    correct =0
    train_bar = tqdm(data_loader_val)
    num = 0
    since = time.time()
    for i, (input, true_target) in enumerate(train_bar):
        input = input.cuda()
        true_target = true_target.cuda()
        
        adv_images = attack.forward(input, true_target)
        adv_images = adv_images.cuda()

        if input_size != val_size:
            resize_adv_images = F.interpolate(input=adv_images, size=val_size, mode='bicubic')
            output = target_model(resize_adv_images)
        else:
            output = target_model(adv_images)
        _, pre = torch.max(output.data, 1)


        name = attack.__class__.__name__
        if num < 5:
            for j in range(true_target.size(0)):
                if pre[j] != true_target[j] and num < 5 :
                    num +=1
                    filename = "%s_%s_%s.png" %(name, args.dataset, str(true_target[j]))
                    load_path = os.path.join(args.output_dir, filename)
                    save_image(torch.stack((input[j], adv_images[j]), 0),  load_path, nrow=1, padding=2, normalize=True, 
                        range=(0,1), scale_each=False, pad_value=0)
                
        total += true_target.size(0)

        correct += (pre != true_target).sum()

        train_bar.set_description("Attack name: {}, dataset: {}, epoch: [{}], asr: {:.4f}".format(args.target_model, args.dataset, i, correct.item() / total *100))
    
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





