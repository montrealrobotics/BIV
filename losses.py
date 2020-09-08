import torch
from torch.nn import Module
from utils import str_to_bool, filter_batch


class IVLoss(Module):
    def __init__(self, epsilon=0.00001, avg_batch = False):


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
        self.avg_batch = avg_batch
        self.epsilon = epsilon       # for numerical stability.
    def forward(self, y_pred,y,lbl_var):
        """
        
        """
        # m = y.shape[0]
        l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y)*(1/(lbl_var + self.epsilon  ) ))
        if self.avg_batch:
            return l/(torch.sum(1/(lbl_var + self.epsilon  ) ))
        # else:
        #     return l



class CutoffMSE(Module):
    def __init__(self, cutoffValue=20):
        super(CutoffMSE, self).__init__()
        self.cutoffValue = cutoffValue
    def forward(self, y_pred,y,lbl_var):
        # Filter the batch samples based on the noise variance
        y_pred, y, lbl_var = filter_batch(y_pred,y,lbl_var,threshold=self.cutoffValue)
        batch_size = y.shape[0]

        if batch_size == 0:
            return "cutoffMSE:NO_LOSS"
        else:
            l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y))

            l = l/batch_size

            return l
    