#!/bin/bash


CKPT="llava-v1.5-7b"
SPLIT="llava_mme"

CKPT_DIR="liuhaotian"
DATA_DIR="../playground/data/eval"


TOKEN=${1}
PARAM="vtn_${TOKEN}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py \
    --experiment ${SPLIT}/${CKPT}/${PARAM}

cd eval_tool

python calculation.py --results_dir answers/${SPLIT}/${CKPT}/${PARAM}


