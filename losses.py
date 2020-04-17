import torch
from torch.nn import Module


class IVLoss(Module):
    def __init__(self, avg_batch = False):
        super(IVLoss, self).__init__()
        self.avg_batch = bool(avg_batch)
        self.epsilon = 1        # for numerical stability.
    def forward(self, y_pred,y,lbl_variance):
        m = y.shape[0]
        l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y)*(1/(lbl_variance+self.epsilon)))
        l = (l/m)
        if self.avg_batch:
            return l/(torch.sum(1/(lbl_variance+self.epsilon)))
        else:
            return l

