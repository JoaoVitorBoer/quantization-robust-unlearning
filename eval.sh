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

conda activate quant
echo "Conda environment: $CONDA_DEFAULT_ENV"

export CUDA_VISIBLE_DEVICES="0,1"
algos=("ga" "ga_gdr" "ga_klr" "npo" "npo_gdr" "npo_klr")
corpuss=("news" "books")
quantize_4bit=0
quantize_8bit=0
use_lora=0
timestamp=$(date +"%Y%m%d_%H%M%S")

print_usage() {
  cat <<'EOF'
Usage: eval.sh [options]
  --corpus "news,books"   Comma- or space-separated list of corpora (default: news books)
  --algo "ga,npo"         Comma- or space-separated list of algos (default: ga ga_gdr ga_klr npo npo_gdr npo_klr)
  --quantize_4bit [0|1]   Enable/disable 4-bit quantization (flag alone sets 1; default: 0)
  --quantize_8bit [0|1]   Enable/disable 8-bit quantization (flag alone sets 1; default: 0)
  --lora                  Read checkpoints from ./baselines/ckpt/lora
  -h, --help              Show this help message
EOF
}

BASE_CKPT_DIR="./baselines/ckpt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus)
      shift
      [[ $# -gt 0 ]] || { echo "Error: --corpus requires a value"; exit 1; }
      value="${1//,/ }"
      read -ra corpuss <<< "$value"
      shift
      ;;
    --algo|--algos)
      shift
      [[ $# -gt 0 ]] || { echo "Error: --algos requires a value"; exit 1; }
      value="${1//,/ }"
      read -ra algos <<< "$value"
      shift
      ;;
    --quantize_4bit|-quantize_4bit)
      if [[ $# -gt 1 && "$2" != -* ]]; then
        quantize_4bit="$2"
        shift 2
      else
        quantize_4bit=1
        shift
      fi
      ;;
    --quantize_8bit|-quantize_8bit)
      if [[ $# -gt 1 && "$2" != -* ]]; then
        quantize_8bit="$2"
        shift 2
      else
        quantize_8bit=1
        shift
      fi
      ;;
    --lora)
      use_lora=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

if [[ $use_lora -eq 1 ]]; then
  BASE_CKPT_DIR="${BASE_CKPT_DIR}/lora"
fi

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

  out_file="results/output_${corpus}_${timestamp}.csv"
  echo "Running eval.py with params:"
  echo "  --model_dirs ${model_dirs[*]}"
  echo "  --names ${names[*]}"
  echo "  --tokenizer_dir meta-llama/Llama-2-7b-hf"
  echo "  --corpus ${corpus}"
  echo "  --quantize_4bit ${quantize_4bit}"
  echo "  --quantize_8bit ${quantize_8bit}"
  echo "  --metrics knowmem_f verbmem_f privleak knowmem_r"
  echo "  --out_file ${out_file}"
  echo -e "---------------------------------------- \n"

  python eval.py \
      --model_dirs "${model_dirs[@]}" \
      --names "${names[@]}" \
      --tokenizer_dir "meta-llama/Llama-2-7b-hf" \
      --corpus "${corpus}" \
      --quantize_4bit "${quantize_4bit}" \
      --quantize_8bit "${quantize_8bit}" \
      --metrics knowmem_f verbmem_f privleak knowmem_r \
      --out_file "${out_file}"
done
