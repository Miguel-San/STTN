import torch

class VGRegularizer:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def build(self, ts:torch.Tensor):
        self.ts = ts
        self.device = ts.device

        self.G = torch.empty(ts.shape[0], ts.shape[0]).to(self.device)
        for i in range(ts.shape[0]):
            for j in range(i):
                self.G[i, j] = self.G[j, i]

            for j in range(i, ts.shape[0]):
                if i == j:
                    self.G[i, j] = 0
                else:
                    self.G[i, j] = self._check_connectivity(i,j)

    def _check_connectivity(self, i, j):      
        idx1, idx2 = min(i, j), max(i, j)
        element_visibility = torch.empty(idx2-idx1).to(self.device)
        element_visibility[0] = 1
        element_visibility[-1] = 1

        visibility_criteria = self.ts[idx2] + (self.ts[idx1] - self.ts[idx2])*(idx2 - torch.arange(idx1+1, idx2, device=self.device))/(idx2 - idx1) - self.ts[idx1+1:idx2]
        element_visibility[1:] = torch.sigmoid(visibility_criteria*self.alpha)

        return torch.min(element_visibility)
    
    @property
    def G(self):
        return self.G