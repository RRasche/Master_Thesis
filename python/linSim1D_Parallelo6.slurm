#!/bin/bash -l
#SBATCH -J o6_Maxwell_Sim
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --mail-type=end
#SBATCH --mail-user=r.rasche@campus.lmu.de
#SBATCH --cpus-per-task=19
#SBATCH --mem-per-cpu=2G

conda activate Sim
cd ~/rasmus/python
srun python linSim1D_Parallelo6.py
