import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from model.llava.mm_utils import tokenizer_image_token

# def collate_fn(
#     batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
# ):
#     image_path_list = []
#     images_list = []
#     images_clip_list = []
#     conversation_list = []
#     masks_list = []
#     label_list = []
#     resize_list = []
#     questions_list = []
#     sampled_classes_list = []
#     offset_list = [0]
#     cnt = 0
#     inferences = []
#     keywords_list = []
#     input_keywords_ids_list = []
    
#     for (
#         image_path,
#         images,
#         images_clip,
#         conversations,
#         masks,
#         label,
#         resize,
#         questions,
#         sampled_classes,
#         inference,
#         keywords,
#     ) in batch:
#         image_path_list.append(image_path)
#         images_list.append(images)
#         images_clip_list.append(images_clip)
#         conversation_list.extend(conversations)
#         label_list.append(label)
#         masks_list.append(masks.float())
#         resize_list.append(resize)
#         questions_list.append(questions)
#         sampled_classes_list.append(sampled_classes)
#         cnt += len(conversations)
#         offset_list.append(cnt)
#         inferences.append(inference)
#         # keywords_list.extend([keywords] * len(conversations)) 
#         keywords_list.append(keywords)

#     if use_mm_start_end:
#         # replace <image> token
#         for i in range(len(conversation_list)):
#             replace_token = DEFAULT_IMAGE_TOKEN
#             replace_token = (
#                 DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             )
#             conversation_list[i] = conversation_list[i].replace(
#                 DEFAULT_IMAGE_TOKEN, replace_token
#             )
#     input_ids = [
#         tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
#         for prompt in conversation_list
#     ]
#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     attention_masks = input_ids.ne(tokenizer.pad_token_id)\
        
#     for keywords in keywords_list:
#         input_keywords_ids = tokenizer(keywords, add_special_tokens=False).input_ids[0]
#         input_keywords_ids_list.append(input_keywords_ids)

#     conv = conversation_lib.default_conversation.copy()
#     targets = input_ids.clone()

#     if conv_type == "llava_v1":
#         sep = conv.sep + conv.roles[1] + ": "
#     else:
#         sep = "[/INST] "
        
#     for conversation, target in zip(conversation_list, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             # if len(parts) != 2:
#             #     break
#             assert len(parts) == 2, (len(parts), rou)
#             parts[0] += sep

#             if DEFAULT_IMAGE_TOKEN in conversation:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         # if False:
#         #     z = target.clone()
#         #     z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
#         #     if local_rank == 0:
#         #         print(
#         #             "conversation: ",
#         #             conversation,
#         #             "tokenizer.decode(z): ",
#         #             tokenizer.decode(z),
#         #         )

#         if cur_len < tokenizer.model_max_length:
#             assert cur_len == total_len

#     if inferences[0] == False:
#         truncate_len = tokenizer.model_max_length - 255

#         if input_ids.shape[1] > truncate_len:
#             input_ids = input_ids[:, :truncate_len]
#             targets = targets[:, :truncate_len]
#             attention_masks = attention_masks[:, :truncate_len]

#     return {
#         "image_paths": image_path_list,
#         "images": torch.stack(images_list, dim=0),
#         "images_clip": torch.stack(images_clip_list, dim=0),
#         "input_ids": input_ids,
#         "labels": targets,
#         "attention_masks": attention_masks,
#         "masks_list": masks_list,
#         "label_list": label_list,
#         "resize_list": resize_list,
#         "offset": torch.LongTensor(offset_list),
#         "questions_list": questions_list,
#         "sampled_classes_list": sampled_classes_list,
#         "input_keywords_ids_list": input_keywords_ids_list,
#         "inference": inferences[0],
#         "conversation_list": conversation_list,
#     }
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
        
        # [KEY]만 추가 (키워드 없이)
        updated_convs = []
        for conv in conversations:
            if "[KEY]" not in conv:  # 중복 방지
                conv = conv.replace("</s>", " [KEY].</s>")
            updated_convs.append(conv)
        conversation_list.extend(updated_convs)
        
        # conversation_list.extend(conversations)
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
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)

        # 각 conversation에 [KEY] 토큰이 있는 keyword를 붙여줌
        updated_convs = []
        for i, conv in enumerate(conversations):
            if keywords[i] is not None:
                conv = conv.replace("</s>", f" [KEY].</s>")
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



class ReasonSegKeyDataset(torch.utils.data.Dataset):
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
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg")
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples:", len(images))

        self.img_to_explanation = {}
        if explanatory != -1:
            with open(os.path.join(base_image_dir, "reason_seg", reason_seg_data, "explanatory_key", "keyword_outputs.json")) as f:
                items = json.load(f)
            for item in items:
                self.img_to_explanation[item["image"]] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                    "keyword": item.get("keyword", None),
                }

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(list(range(len(sents))), size=self.num_classes_per_sample, replace=False)
        else:
            sampled_inds = list(range(len(sents)))

        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [(mask == 1).astype(np.float32) for _ in range(len(sampled_inds))]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image_name = image_path.split("/")[-1]

        # choice 결정
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            choice = 2 if random.random() < self.explanatory else random.randint(0, 1)
        else:
            choice = 0

        questions, answers, keywords = [], [], []
        for text in sampled_sents:
            question_template = random.choice(self.long_question_list if is_sentence else self.short_question_list)
            questions.append(question_template.format(sent=text) if is_sentence else question_template.format(class_name=text.lower()))

            keyword = None
            if self.explanatory != -1 and image_name in self.img_to_explanation:
                keyword = self.img_to_explanation[image_name].get("keyword", None)
                if choice == 0:
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " " + answer
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text + " " + random.choice(EXPLANATORY_QUESTION_LIST)
                    answers.append(answer)
                elif choice == 2:
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
            else:
                answers.append(random.choice(self.answer_list))

            keywords.append(keyword)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for i in range(len(questions)):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = torch.from_numpy(np.stack(sampled_masks, axis=0))
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            False,
            keywords,
        )
        
        
from .refer import REFER
from pycocotools import mask

class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        # elif len(splits) == 3:
        #     ds, splitBy, split = splits
        #     self.base_image_dir = os.path.join(self.base_image_dir,'refer_seg')
        #     refer_api = REFER(self.base_image_dir, ds, splitBy)
        #     ref_ids_val = refer_api.getRefIds(split=split)
        #     images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
        #     refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
        #     refer_seg_ds = {}
        #     refer_seg_ds["images"] = []
        #     loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
        #     for item in loaded_images:
        #         item = item.copy()
        #         if ds == "refclef":
        #             item["file_name"] = os.path.join(
        #                 base_image_dir, "images/saiapr_tc-12", item["file_name"]
        #             )
        #         elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
        #             item["file_name"] = os.path.join(
        #                 self.base_image_dir,
        #                 "images/mscoco/images/train2014",
        #                 item["file_name"],
        #             )
        #         refer_seg_ds["images"].append(item)
        #     refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

        #     img2refs = {}
        #     for ref in refs_val:
        #         image_id = ref["image_id"]
        #         img2refs[image_id] = img2refs.get(image_id, []) + [
        #             ref,
        #         ]
        #     refer_seg_ds["img2refs"] = img2refs
        #     self.refer_seg_ds = refer_seg_ds
        #     self.data_type = "refer_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            self.base_image_dir = os.path.join(self.base_image_dir, 'refer_seg')
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)

            # 하나의 샘플 = 하나의 ref 표현
            self.images = refs_val  # ✅ reason_seg와 구조 맞춤
            self.refer_seg_ds = {
                "images": [],  # mscoco style image dicts
                "annotations": refer_api.Anns,
                "img2refs": {}
            }

            loaded_images = refer_api.loadImgs(image_ids=refer_api.getImgIds(ref_ids=ref_ids_val))
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        self.base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        self.base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                self.refer_seg_ds["images"].append(item)

            for ref in refs_val:
                image_id = ref["image_id"]
                self.refer_seg_ds["img2refs"].setdefault(image_id, []).append(ref)

            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )