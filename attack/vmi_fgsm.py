import torch

class VMI_FGSM(object):
    def __init__(self, model,loss, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=20, beta=3/2):
        self.model = model
        self.loss = loss
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.N = N
        self.beta = beta
        self.device = next(model.parameters()).device

    def forward(self, images, labels):
        momentum = torch.zeros_like(images).detach().to(self.device)
        v = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        

        for i in range(self.steps):
            adv_images.requires_grad = True            
            output = self.model(adv_images)
            cost = self.loss(output, labels)

            adv_grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v),
                                     dim=(1, 2, 3), keepdim=True)
            grad = grad+ momentum * self.decay
            momentum =grad

            # Calculate Gradient Varience
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.rand_like(images).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                output = self.model(neighbor_images)
                cost = self.loss(output, labels)

                GV_grad += torch.autograd.grad(cost, neighbor_images, retain_graph=False, create_graph=False)[0]

            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, 
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images+ delta, min=0, max=1).detach()

        return adv_images






