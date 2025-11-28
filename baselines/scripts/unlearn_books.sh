#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1                        # Number of nodes to use
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --mem=20G                        # Memory per node (e.g., 16 gigabytes)
#SBATCH --time=2-00:00:00                  # Wall-clock time limit (HH:MM:SS)
#SBATCH --gpus=2

# algos available choice=("npo_gdr" "npo_klr" "ga_gdr" "ga" "ga_klr" "npo")
# alpha is for utility constraint, threshold is for filtering out salient modules

cd /home/joaoabitante/quantization-robust-unlearning/baselines

conda activate quant
echo "Conda environment: $CONDA_DEFAULT_ENV"

export CUDA_VISIBLE_DEVICES="0,1"
CORPUS=('books')
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR='muse-bench/MUSE-Books_target'
LLAMA_DIR='meta-llama/Llama-2-7b-hf'
MAX_LEN=2048
EPOCHS=5
LR='1e-5'
PER_DEVICE_BATCH_SIZE=2
algos=("npo_gdr" "npo_klr" "ga_gdr" "ga_klr" "ga" "npo")
alphas=(300 2 100 2 1 1) # last two are for ga and npo but they are just placeholders and will not be used

for i in "${!algos[@]}"; do
    algo="${algos[$i]}"
    alpha="${alphas[$i]}"
    echo "===== Starting unlearning run: $algo epochs=$EPOCHS lr=$LR  ====="
    python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --alpha "$alpha" \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE
    echo -e "\n===== Finished unlearning run: $algo ===== \n\n\n"
done
