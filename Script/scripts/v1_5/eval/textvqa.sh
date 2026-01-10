#!/bin/bash

CKPT_DIR="liuhaotian"
DATA_DIR="../playground/data/eval"

CKPT="llava-v1.5-7b"
SPLIT="llava_textvqa_val_v051_ocr"

TOKEN=${1}
PARAM="vtn_${TOKEN}"

# python -m llava.eval.model_vqa_loader \
#     --model-path ${CKPT_DIR}/${CKPT} \
#     --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
#     --image-folder ${DATA_DIR}/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
#     --visual_token_num ${TOKEN} \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# echo "Result file: ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl
