from torchvision import models
import timm
import torchvision.transforms as transforms



def load_transform(name, args):
    input_size = 229
    if args.dataset == 'imagenet':
        if name == 'resnet101':
            transform =  transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_size = 224
        elif name in ['inc_v3', 'inc_v4','inc_res_v2', 'inc_res_v2_ens', 'inc_v3_adv']:
            transform =  transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_size = 229
        else:
            raise Exception("Surrogate_model name must be in [inc_v3, inc_v3_adv, inc_res_v2, inc_res_v2_ens, inc_v4, resnet101]")
    else:
        raise Exception("Dataset must be imagenet!!!")
    
    return transform, input_size


def load_surrogate_model(name):
    if name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif name == 'inc_v4':
        model = timm.create_model('inception_v4', pretrained=True)
    elif name == 'inc_res_v2':
        model = timm.create_model('inception_resnet_v2', pretrained=True)
    elif name == 'inc_res_v2_ens':
        model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=True)
    elif name == 'inc_v3':
        model = models.inception_v3(pretrained=True)
    elif name == 'inc_v3_adv':
        model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    else:
        raise Exception("Surrogate_model name must be in [inc_v3, inc_v3_adv, inc_res_v2, inc_res_v2_ens, inc_v4, resnet101]")

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


    
