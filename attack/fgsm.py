import torch

class FGSM(object):
    def __init__(self, model, loss, eps):
        self.model = model
        self.loss = loss
        self.eps = eps

    
    def forward(self, images, labels):
        images.requires_grad = True

        outputs = self.model(images)

        self.model.zero_grad()

        cost = self.loss(outputs, labels).cuda()
        cost.backward()

        adv_images = images + self.eps * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images
