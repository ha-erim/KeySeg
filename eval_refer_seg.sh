#!/bin/bash

# 사용할 GPU 설정 (2번, 3번)
export CUDA_VISIBLE_DEVICES=2,3

# 실행 옵션 설정
MODEL_PATH="/home/hrkim/LISA/runs/key_fusion_ce_es_upsamle_sa_tmd_ref_200/hf"
VISION_TOWER="openai/clip-vit-large-patch14"
IMAGE_FOLDER="/home/hrkim/dataset/refer_seg/"
SAM_PATH="/home/hrkim/dataset/sam_vit_h_4b8939.pth"
SAVE_DIR="./eval/ref_seg_results"
# DATASET_SPLIT="refcocog|val"
CONV_MODE="llava_v1"

# 실행
python eval_refer_seg.py \
  --model-path $MODEL_PATH \
  --vision-tower $VISION_TOWER \
  --image-folder $IMAGE_FOLDER \
  --sam_path $SAM_PATH \
  --save_file $SAVE_DIR \
  --conv-mode $CONV_MODE \
  --precision bf16 \
  --model_max_length 512 \
  --temperature 0.2 \
  --max_new_tokens 1024 \
  --num-chunks 1 \
  --chunk-idx 0
