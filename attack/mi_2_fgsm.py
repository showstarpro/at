import torch

class MI_2_FGSM(object):
    def __init__(self, model_1, model_2,loss, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        self.model_1 = model_1
        self.model_2 = model_2
        self.loss = loss
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.device = next(model_1.parameters()).device

    def forward(self, images, labels):
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        
        for i in range(self.steps):
            adv_images.requires_grad = True            
            output_1 = self.model_1(adv_images)
            output_2 = self.model_2(adv_images)
            cost_1 = self.loss(output_1, labels)
            cost_2 = self.loss(output_2, labels)
            grad_1 = torch.autograd.grad(cost_1, adv_images, retain_graph=False, create_graph=False)[0]

            grad_2 = torch.autograd.grad(cost_2, adv_images, retain_graph=False, create_graph=False)[0]

            grad = grad_1 + grad_2
            grad = grad / torch.mean(torch.abs(grad),
                                     dim=(1, 2, 3), keepdim=True)
            grad = grad+ momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, 
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images+ delta, min=0, max=1).detach()

        return adv_images






