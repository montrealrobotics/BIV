import torch
from torch.nn import Module


class IVLoss(Module):
    def __init__(self):
        super(IVLoss, self).__init__()
    
    def forward(self, y_pred,y,lbl_variance):
        m = y.shape[0]
        l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y)*(1/lbl_variance))
        return (l/m)
