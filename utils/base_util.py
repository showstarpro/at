import torchvision.transforms as transforms

mean = (0.48145466,0.4578275,0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm_transforms = transforms.Normalize(mean,std)
def norm_image(image):
    return norm_transforms(image)