import torch

class NI_FGSM(object):
    def __init__(self, model,loss, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        self.model = model
        self.loss = loss
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.device = next(model.parameters()).device

    def forward(self, images, labels):
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        

        for i in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum            
            output = self.model(nes_images)
            cost = self.loss(output, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            grad = self.decay * momentum + grad / torch.mean(torch.abs(grad),
                                     dim=(1, 2, 3), keepdim=True)
            momentum =grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, 
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images+ delta, min=0, max=1).detach()

        return adv_images






