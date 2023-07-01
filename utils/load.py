from torchvision import models
import timm
import torchvision.transforms as transforms



def load_transform(name=None, args=None):
    input_size = 229
    if args.dataset == 'imagenet':
        # if name == 'resnet101':
        #     transform =  transforms.Compose([
        #                 transforms.Resize(256, interpolation=3),
        #                 transforms.CenterCrop(224),
        #                 transforms.ToTensor(),
        #                 # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #             ])
        #     input_size = 224
        # elif name in ['inc_v3', 'inc_v4','inc_res_v2', 'inc_res_v2_ens', 'inc_v3_adv']:
        #     transform =  transforms.Compose([
        #                 transforms.Resize(299),
        #                 transforms.CenterCrop(299),
        #                 transforms.ToTensor(),
        #                 # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #             ])
        #     input_size = 229
        # else:
        #     raise Exception("Surrogate_model name must be in [inc_v3, inc_v3_adv, inc_res_v2, inc_res_v2_ens, inc_v4, resnet101]")
        transform =  transforms.Compose([
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        input_size = 224
    else:
        raise Exception("Dataset must be imagenet!!!")
    
    return transform, input_size


def load_model(model_name):
    # inception_v4  inception_resnet_v2  inception_v3 inception_v3.tf_adv_in1k
    # resnet101 resnet18 resnet50  vgg16_bn  googlenet
    timm_model_list = ['inception_v4','inception_resnet_v2', 'inception_v3.tf_adv_in1k']
    vision_model_list = ['resnet101','vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16'
    ]
    if model_name in timm_model_list:
        model = timm.create_model(model_name, pretrained=True)
    elif model_name in vision_model_list:
        model = getattr(models, model_name)(pretrained=True)
    else:
        raise Exception(f"Surrogate_model name: {model_name} must be in {timm_model_list} or {vision_model_list}")

    return model
    

def load_target_model(name):
    val_size = 299
    if name == 'inc_v3':
        target_model = models.inception_v3(pretrained=True)     
    elif name == 'inc_v4':
        target_model = timm.create_model('inception_v4', pretrained=True)
    elif name == 'inc_res_v2':
        target_model = timm.create_model('inception_resnet_v2', pretrained=True)
    elif name == 'inc_res_v2_ens':
        target_model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=True)
    elif name == 'inc_v3_adv':
        target_model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    elif name == 'resnet101':
        target_model = models.resnet101(pretrained=True)
        val_size = 224
    else:
        raise Exception("Target_model name must be in [inc_v3, inc_v3_adv, inc_res_v2, inc_res_v2_ens, inc_v4, resnet101]")
    
    return target_model, val_size


    
