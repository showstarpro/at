import torch

class PGD(object):
    def __init__(self, model,loss, eps=8/255, alpha=2/255, steps=40):
        self.model = model
        self.loss = loss
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

        self.device = next(model.parameters()).device

    def forward(self, images, labels):

        ori_images = images.data

        for i in range(self.steps):
            images.requires_grad = True
            ouputs = self.model(images)

            self.model.zero_grad()
            cost = self.loss(ouputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha * images.grad.sign()

            delta = torch.clamp(adv_images - ori_images, min = -self.eps, max= self.eps)
            images = torch.clamp(ori_images + delta, min=0, max=1).detach_()

        return images






