#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="liuhaotian"
DATA_DIR="../playground/data/eval"

CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"


TOKEN=${1}
PARAM="vtn_${TOKEN}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

GQA_DIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst ${GQA_DIR}/testdev_balanced_predictions.json

cd ${GQA_DIR}
python eval/eval.py --tier testdev_balanced
