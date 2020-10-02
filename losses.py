import torch
from torch.nn import Module
from utils import str_to_bool, filter_batch


class BIVLoss(Module):
    def __init__(self, epsilon=0.00001):


        """
        Description:
            Batch Inverse variance loss is defined as:
            
        .. math::
            
            \mathcal{L}_{\mathrm{batch}}(D_i, \\theta) = \left( \sum_{k=0}^K  \dfrac{1}{\sigma_k^2 + \epsilon}  \\right)^{-1} \sum_{k=0}^K \dfrac{\mathcal{L}\left(f(x_k, \\theta),\overline{y}_k\\right)}{\sigma_k^2 + \epsilon}
            \label{eq:IVweighted_loss}

            """

        super(BIVLoss, self).__init__()
        self.epsilon = epsilon       # for numerical stability.
    def forward(self, y_pred,y,lbl_var):
        """
        Description:
            Compute the forward pass for BIV loss function.
        """
        # m = y.shape[0]
        l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y)*(1/(lbl_var + self.epsilon  ) ))
        return l/(torch.sum(1/(lbl_var + self.epsilon  ) ))


class CutoffMSE(Module):
    def __init__(self, cutoffValue=1):
        """
        Description:

            Cutoff MSE loss has two steps:
            
            1) Filtering:
                Filter the noisy labels out by comparing their noise variances to a threshold. Remove the label if its noise variance is bigger than a threshold, otherwise, use it to compute the loss.
            
            2) MSE loss: 
                Compute the mse loss using the filtered labels.
        
        Args:
            :threshold: An upper bound noise variance threshold to filter out the noisy labels.
        """

        super(CutoffMSE, self).__init__()
        self.cutoffValue = cutoffValue

    def forward(self, y_pred,y,lbl_var):
        """
        Description:
            Compute the forward pass.
        """

        y_pred, y, lbl_var = filter_batch(y_pred,y,lbl_var,threshold=self.cutoffValue)
        batch_size = y.shape[0]

        if batch_size == 0:
            return "cutoffMSE:NO_LOSS"
        else:
            l = torch.matmul(torch.sub(y_pred,y).t(), torch.sub(y_pred,y))

            l = l/batch_size

            return l
    