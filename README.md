# Inverse Variance Weighting for Heteroscedastic Noisy Labels in Deep Learning

## Introduction

One of the fundamental  assumptions  in  super-vised deep learning is that the labels are correct. However, this assumption does not hold in many cases.  Error (noise) on the labels can severely impact the ability of the model to learn. In many cases, the noise on the labels is heteroscedastic and the variable variance is either known, or can be readily estimated from analysis on the dataset. In this work, we propose to leverage the known label variance as a tool to improve learning. Intuitively, we use an information theoretic approach and weight training samples by the inverse variance in the training loss function. We also show that weight normalization by mini-batch (rather than at the level of the entire dataset) improves stability in the learning process. 

### Prerequisites

To run the code, we wrapped up all the libraries inside a singularity container, you can download it [here](). To manually build your environment using anaconda, we provide yml file [here](). 

### Run the Code

To run the code:

```bash
python main.py --exp_settings="classical_1_7159,7159,True,mse,5000" --noise_settings="True,uniform,True,False,0.5,1,False,3" \\
--noise_params="0,0,0,0"  --estim_noise_params="0.52,500,0.09,0"
```

To run the code locally inside singularity container:

```bash
singularity exec --nv -H $HOME:/home/ -B $SLURM_TMPDIR:/datasets/ \\
-B $SLURM_TMPDIR:/final_outps/  $SLURM_TMPDIR/pytorch_f.simg python ~/apps/IV_RL_server/main.py \\
--exp_settings=$1 --noise_settings=$2 --noise_params=$3 --estim_noise_params=$4
```



To run the code in a cluster that supporting Slurm workload manager, use this starter code:

```bash
#!/bin/bash
#SBATCH -o /home/[organization]/k/[username]/logs/noise_%j.out   # Change this!
#SBATCH --cpus-per-task=4  
#SBATCH --gres=gpu:1        
#SBATCH --mem=32Gb    

# Load cuda
module load cuda/10.0
# 1. You have to load singularity
module load singularity
# 2. Then you copy the container to the local disk
rsync -avz /home/[organization]/k/[username]/environments/pytorch_f.simg $SLURM_TMPDIR     # Change this!
# 3. Copy your dataset on the compute node
rsync -avz /network/tmp1/[username]/ $SLURM_TMPDIR        # Change this!
# 3.1 export wandb api key
export WANDB_API_KEY= "put your wandb key here"       # Change this!
# 4. Executing your code with singularity
singularity exec --nv -H $HOME:/home/ -B $SLURM_TMPDIR:/datasets/ -B $SLURM_TMPDIR:/final_outps/  $SLURM_TMPDIR/pytorch_f.simg python ~/apps/IV_RL_server/main.py --exp_settings=$1 --noise_settings=$2 --noise_params=$3 --estim_noise_params=$4
# 5. Move results back to the login node.
rsync -avz $SLURM_TMPDIR --exclude="Datasets" --exclude="pytorch_f.simg"  /home/[organization]/k/[username]/outputs  # Change this!
```



## [Explanation] Command-line Arguments

- **--exp_settings:** These are the arguments that controlling whole the experiment.

  - **Tag** : A wandb tag: ***str***

  - **Seed**: Experiment seed : ***int***

  - **Normalisation**: data normalisation : ***boolean***

  - **Loss type**: Loss function type: ***str***

    - **Mean squared error**: **mse**
    -  **Inverse variance**: **iv**
    -  **batch Inverse variance**: **biv**

  - **Noise average mean**: Average over distributions mean

    - > average mean = p*mu1 + (1-p)*mu2
      >
      > p = flip-coin probability
      >
      > mu1 = mean of the first distribution.
      >
      > mu2 = mean of the second distribution.

- **--noise_settings:** The arguments that are controlling the noises that are being added to the data.

  - **Noise**: add the noise or not: ***bool***
  - **Noise type:** type of the noise: ***str***
    - **uniform distribution:** "uniform" 
    - **Gamma distribution** : "gamma" 
  - **Noise parameters estimation** : Enable parameters estimation of the noise distributions, rather than providing them directly: ***bool***
  - **Maximum heteroscedasticty**: ***bool***
  - **heteroscedasticty scale** : Scale the maximum heteroscedasticty value with an scalar: ***float***
  - **Flip-coin**: random Probability over noise distributions: ***float***
  - **Noise threshold** : Noise cutoff threshold: ***bool***
  - **Threshold value** : Threshold cutoff value: ***float***

- **--noise_params:**  Noise distribution parameters,  i.e  a and b in the case of uniform distribution, you can pass as many parameters you want: ***str***

- **--estim_noise_params:** A list of  mu and v to estimate the parameters of the noise distributions, in this case you should pass the mean (mu) and the variance (v) of the noise distributions: ***str***

## Contributors

* **Waleed Khamies**
* **Vincent Mai**

## License

This project is licensed under the [???] License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement

* [Liam Paull](https://liampaull.ca/) - Principle Investigator - [Robotics Lab (UdeM University)](https://montrealrobotics.ca/)
* [National Sciences and Engineering Research Council of Canada](https://www.nserc-crsng.gc.ca/) 
 
