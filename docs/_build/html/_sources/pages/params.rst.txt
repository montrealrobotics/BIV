params.py
==========

This is the main file to configure all the code base, you can configure:
    - Dataloaders.
    - Models.
    - Training parameters.

Keep in mind that there are another parameters that need to be specified through the command line, as they are considered the most important paramters in our experiments, they are:

    - :math:`\mu` and V of uniform noise. (float)
    - IV batch normalization. (boolean)
    - Experiment tag. (string)