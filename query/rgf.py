import torch
import utils

class RGF:
    def __init__(self, model, loss, q, sigma):
        self.model = model
        self.loss = loss
        self.q = q
        self.sigma = sigma


    def query(self, image, label):
        cur_image = image.detach()
        with torch.no_grad():
            output = self.model(utils.norm_image(cur_image))
            cost = self.loss(output, label)

        # random sample vectors
        us = [] # random vectors
        for _ in range(self.q):
            us.append(torch.randn_like(cur_image).cpu())
        
        # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
        orthos = []
        for u in us:
            for ou in orthos:
                u = u - (torch.sum(u * ou,dim=[1,2,3]))[:,None,None,None] * ou
            u = u / torch.sqrt(torch.sum(u * u,dim=[1,2,3]))[:,None,None,None]
            orthos.append(u)
            # torch.linalg.qr()

        g = 0
        with torch.no_grad():
            for u in orthos:
                u_cpu = u.clone().type_as(cur_image)
                output =self.model(utils.norm_image(cur_image + self.sigma * u_cpu))
                cost_q = self.loss(output, label)
                g = g + (u_cpu * (cost_q - cost) / self.sigma)

        return g






        
        








    