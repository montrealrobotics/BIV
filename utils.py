import torch




def get_unif_Vmax(mu, scale_value =1):
     
    """"
    Description:
        Estimates the maximum variance needed for the uniform distribution to produce maximum heteroscedasticity.

    Return:
        :vmax: the maximum uniform variance.
    Return type:
        float
    Args:
        :mu: mean of the the uniform.
        :scale_value [optional]: controls scaling the Vmax to different values.
    
    Here is the formula for estimating :math:`V_{max}`

    .. math::
        V_{max} = \\frac{4 \mu^2}{12}

    """
    vmax = (4*mu**2)/12
    vmax = vmax/scale

    return vmax