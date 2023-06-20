import torch


class RGF:
    def __init__(self, model, loss, q, sigma):
        self.model = model
        self.loss = loss
        self.q = q
        self.sigma = sigma


    def query(self, image, label):
        cur_image = image.detach()
        output = self.model(cur_image)
        cost = self.loss(output, label)

        # random sample vectors
        us = [] # random vectors
        for _ in range(self.q):
            us.append(torch.randn_like(cur_image))
        
        # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
        orthos = []
        for u in us:
            for ou in orthos:
                u = u - torch.sum(u * ou) * ou
            u = u / torch.sqrt(torch.sum(u * u))
            orthos.append(u)

        images = []
        for u in orthos:
            images.append(cur_image + self.sigma * u)
        images = torch.cat(images)

        outputs = self.model(images)

        g = 0
        for i in range(self.q):
            cost_q = self.loss(outputs[i].unsqueeze(0), label)
            g += orthos[i] * (cost_q - cost) / self.sigma

        return g






        
        








    