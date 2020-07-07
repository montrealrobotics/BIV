# Inverse Variance Weighting for Heteroscedastic Noisy Labels in Deep Learning

## Introduction

One  of  the fundamental  assumptions  in  super-vised deep learning is that the labels are correct. However, this assumption does not hold in many cases.  Error (noise) on the labels can severely impact the ability of the model to learn. In many cases, the noise on the labels is heteroscedastic and the variable variance is either known, or can be readily estimated from analysis on the dataset. In this work, we propose to leverage the known label variance as a tool to improve learning. Intuitively, we use an information theoretic approach and weight training samples by the inverse variance in the training loss function. We also show that weight normalization by mini-batch (rather than at the level of the entire dataset) improves stability in the learning process. 

### Prerequisites

To run the code, we wrapped up all the libraries inside a singularity container, you can download it [here](). To manually build your environment using anaconda, we provide yml file [here](). 

### Run the Code

To run the code:

```bash
python main.py --exp_settings="hello,42,False,mse, 5000" \\
--noise_settings="True,uniform,True,False,0.5,0.5,False,3" \\
--noise_params="1,100,100,1000" --estim_noise_params="2,500,1,0"
```

To run the code inside singularity container:

```bash
singularity exec --nv -H $HOME:/home/ -B $SLURM_TMPDIR:/datasets/ \\
-B $SLURM_TMPDIR:/final_outps/  $SLURM_TMPDIR/pytorch_f.simg python ~/apps/IV_RL_server/main.py \\
--exp_settings=$1 --noise_settings=$2 --noise_params=$3 --estim_noise_params=$4
```

## Command-line Arguments

- **--exp_settings:** The arguments that are controlling whole the experiment.

  - Tag : a wandb tag: ***str***

  - Seed: experiment seed : ***int***

  - Normalisation: data normalisation : ***boolean***

  - Loss type: loss function: ***str***

    - Mean squared error: **mse**
    -  Inverse variance: **iv**
    -  batch Inverse variance: **biv**

  - Noise average mean: average over distributions mean

    - > average mean = p*mu1 + (1-p)*mu2
      >
      > p = flip-coin probability
      >
      > mu1 = mean of the first distribution.
      >
      > mu2 = mean of the second distribution.

- **--noise_settings:** The arguments that are controlling the noises that are being added to the data.

  - Noise: add the noise or not: ***bool***
  - Noise type: uniform or gamma : ***str***
  - Noise parameters estimation : estimate the noise distribution parameters rather than providing them directly: ***bool***
  - Maximum heteroscedasticty: ***bool***
  - heteroscedasticty scale : scale the maximum heteroscedasticty value with an scalar: ***float***
  - Flip-coin: random probability over noise distributions: ***float***
  - Noise threshold : noise cutoff threshold: ***bool***
  - Threshold value : threshold cutoff value: ***float***

- **--noise_params:**  Noise distribution parameters,  i.e  a and b in the case of uniform distribution, you can pass as many parameters you want: ***str***

- **--estim_noise_params:** Noise distribution estimated parameters, in this case you should pass the mean (mu) and the variance (v) of the noise distribution: ***str***

## Contributors

* **Waleed Khamies**
* **Vincent Mai**

## License

This project is licensed under the [???] License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement

* [Liam Paull](https://liampaull.ca/) - Principle Investigator - [Robotics Lab (UdeM University)](https://montrealrobotics.ca/)

