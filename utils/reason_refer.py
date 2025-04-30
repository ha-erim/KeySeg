import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .reason_seg_key_dataset import ReasonSegKeyDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def collate_fn_val(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    offset_list = [0]
    cnt = 0
    inferences = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        _,
        _,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    truncate_len = tokenizer.model_max_length - 255
    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": None,
        "sampled_classes_list": None,
        "input_keywords_ids_list": [[] for _ in conversation_list],  # 비워줌
        "inference": True,
        "conversation_list": conversation_list,
    }
    
def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    keywords_list = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
        keywords,
        _,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)

        # 각 conversation에 [KEY] 토큰이 있는 keyword를 붙여줌
        updated_convs = []
        for i, conv in enumerate(conversations):
            if keywords[i] is not None:
                conv = conv.replace("</s>", f"[KEY] {keywords[i]}</s>")
            updated_convs.append(conv)
            keywords_list.append(keywords[i])  # keyword 1개씩 누적

        conversation_list.extend(updated_convs)

        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    # 이미지 토큰 <image> 처리
    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # Prompt -> Token
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # keywords -> input_ids 형태로 저장
    input_keywords_ids_list = []
    for keyword in keywords_list:
        if keyword is None:
            input_keywords_ids_list.append([])
        else:
            ids = tokenizer(keyword, add_special_tokens=False).input_ids
            input_keywords_ids_list.append(ids)

    # Target 생성
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    # 너무 길면 자름
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "input_keywords_ids_list": input_keywords_ids_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="refer_seg||reason_seg",
        sample_rate=[9, 3],
        refer_seg_data="refcocog",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegKeyDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference