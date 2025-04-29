#!/bin/bash
#BATCH -J lightning_rnn
#SBATCH --partition=titan
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00        # Time limit hrs:min:sec
#SBATCH --mem=4gb
#SBATCH -o run_rnn_%j.out
#SBATCH -e run_rnn_%j.err
source /Users/aysa7458/DevEnvironment/bin/activate
for version in $(seq 903.0 906.0)
do
python3 nn_lightning.py gru --gpus 1 --batch_size 16 --learning_rate 0.00001 --epochs 1 --grad_clip 0.05 --dropout 0.02 --n_classes 11 --downsample 1 --version $version --data_dir /Users/aysa7458/FireflyClassification/data/dataset_data/
done
