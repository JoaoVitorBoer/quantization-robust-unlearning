#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1                        # Number of nodes to use
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --mem=20G                        # Memory per node (e.g., 16 gigabytes)
#SBATCH --time=2-00:00:00                  # Wall-clock time limit (HH:MM:SS)
#SBATCH --gpus=2

# set `quantize_4bit=0, quantize_8bit=0` to test model in full precision. 
# `quantize_4bit=1, quantize_8bit=0` to test model in 4-bit. 
# `quantize_4bit=0, quantize_8bit=1` to test model in 8-bit

export CUDA_VISIBLE_DEVICES="0,1"
algos=("ga_gdr" "npo_gdr" "npo_klr") 
corpuss=("news")
quantize_4bit=0
quantize_8bit=0

BASE_CKPT_DIR="./baselines/ckpt"

for corpus in "${corpuss[@]}"; do
  # build array paths for --model_dirs e --names
  model_dirs=()
  names=()

  for algo in "${algos[@]}"; do
    model_dirs+=("${BASE_CKPT_DIR}/${corpus}/${algo}")
    names+=("${algo}")
  done

  echo "Corpus: ${corpus}"
  echo "Model dirs: ${model_dirs[@]}"
  echo "Names:      ${names[@]}"

  python eval.py \
      --model_dirs "${model_dirs[@]}" \
      --names "${names[@]}" \
      --tokenizer_dir "meta-llama/Llama-2-7b-hf" \
      --corpus "${corpus}" \
      --quantize_4bit "${quantize_4bit}" \
      --quantize_8bit "${quantize_8bit}" \
      --metrics knowmem_f verbmem_f privleak knowmem_r \
      --out_file "output_${corpus}.csv"
done
