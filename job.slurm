#!/bin/bash

#SBATCH --job-name=python-script
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mail-user=john.wilburn@my.utsa.edu
#SBATCH --mail-type=ALL
#SBATCH --output "output.txt"
#SBATCH --partition "gpu1v100"

python3 main.py > output.txt