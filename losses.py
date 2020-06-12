import torch
from torch.nn import Module


class IVLoss(Module):
    def __init__(self, avg_batch = False):


        """
        Description:
            Inverse variance loss is defined as:
            
        .. math::

            
            L(\overline{y},y,\sigma) = \\frac{ \sum_{i=0}^{m} (\overline{y_i} - y_i)^2 * \sigma_i }{m} 

        Or with batch normalization:

        .. math::

            
            L(\overline{y},y,\sigma) &= \\frac{ \sum_{i=0}^{m} (\overline{y_i} - y_i)^2 * \sigma_i }{m*w} 

            w &= \\frac{1}{\sum_{i=0}^{m} \sigma_i}

            
            """

        super(IVLoss, self).__init__()
        self.avg_batch = bool(avg_batch)
        self.epsilon = 1        # for numerical stability.
    def forward(self, y_pred,y,lbl_var):
        """
        
        """
        m = y.shape[0]
        l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y)*(1/(lbl_var + self.epsilon  ) ))
        if self.avg_batch:
            return l/(torch.sum(1/(lbl_var + self.epsilon  ) ))
        else:
            return l

