import os
import random
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
from model.llava.constants import DEFAULT_IMAGE_TOKEN

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .refer import REFER

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


class ReferSegValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, tokenizer, vision_tower, val_dataset, image_size=1024):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")

        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"

        elif len(splits) == 3:
            ds, splitBy, split = splits
            self.base_image_dir = os.path.join(self.base_image_dir, 'refer_seg')
            refer_api = REFER(self.base_image_dir, ds, splitBy)

            # refs와 이미지 매핑
            self.refs = refer_api.loadRefs(ref_ids=refer_api.getRefIds(split=split))
            image_ids = list(set(ref["image_id"] for ref in self.refs))  # ✅ image_id 커버 완전하게
            loaded_images = refer_api.loadImgs(image_ids=image_ids)
            self.imgs = {img["id"]: img for img in loaded_images}

            # 🛡️ KeyError 방지: 존재하지 않는 image_id 제거
            missing_ids = [ref["image_id"] for ref in self.refs if ref["image_id"] not in self.imgs]
            if len(missing_ids) > 0:
                print(f"[Missing] {len(missing_ids)} image_ids: showing first 5: {missing_ids[:5]}")
                self.refs = [ref for ref in self.refs if ref["image_id"] in self.imgs]

            self.annotations = refer_api.Anns
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refs)
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            ref = self.refs[idx]
            image_id = ref["image_id"]
            image_info = self.imgs[image_id]
            image_path = image_info["file_name"]

            if self.ds == "refclef":
                image_path = os.path.join(
                    self.base_image_dir, "images/saiapr_tc-12", image_info["file_name"]
                )
            else:
                image_path = os.path.join(
                    self.base_image_dir,
                    "images/mscoco/images/train2014",
                    image_info["file_name"],
                )

            sents = [s["sent"].strip().lower() for s in ref["sentences"]]
            ann_id = ref["ann_id"]
            sampled_sents = [sents[0]]
            sampled_ann_ids = [ann_id]

        else:  # reason_seg

            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, _ = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # conversation
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for text in sampled_sents:
            conv.messages = []
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
            conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())

        if self.data_type == "refer_seg":
            ann = self.annotations[ann_id]
            if not ann["segmentation"]:
                m = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            else:
                if isinstance(ann["segmentation"][0], list):
                    rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                else:
                    rle = ann["segmentation"]
                    for r in rle:
                        if not isinstance(r["counts"], bytes):
                            r["counts"] = r["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(m, axis=2).astype(np.uint8)
            masks = torch.from_numpy(np.expand_dims(m, axis=0))
        else:
            masks = torch.from_numpy(np.expand_dims(mask_json, axis=0))

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


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

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

    # def __getitem__(self, idx):
    #     ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
    #     ds = self.refer_seg_ds_list[ds]
    #     refer_seg_ds = self.refer_seg_data[ds]
    #     images = refer_seg_ds["images"]
    #     annotations = refer_seg_ds["annotations"]
    #     img2refs = refer_seg_ds["img2refs"]
    #     idx = random.randint(0, len(images) - 1)
    #     image_info = images[idx]
    #     image_path = image_info["file_name"]
    #     image_id = image_info["id"]
    #     refs = img2refs[image_id]
    #     if len(refs) == 0:
    #         return self.__getitem__(0)

    #     sents = []
    #     ann_ids = []
    #     for ref in refs:
    #         for sent in ref["sentences"]:
    #             text = sent["sent"]
    #             sents.append(text)
    #             ann_ids.append(ref["ann_id"])
    #     if len(sents) >= self.num_classes_per_sample:
    #         sampled_inds = np.random.choice(
    #             list(range(len(sents))), size=self.num_classes_per_sample, replace=False
    #         )
    #     else:
    #         sampled_inds = list(range(len(sents)))
    #     sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
    #     # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
    #     sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
    #     sampled_classes = sampled_sents
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # preprocess image for clip
    #     image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
    #         "pixel_values"
    #     ][0]

    #     image = self.transform.apply_image(image)  # preprocess image for sam
    #     resize = image.shape[:2]

    #     questions = []
    #     answers = []
    #     for text in sampled_classes:
    #         text = text.strip()
    #         assert len(text.split("||")) == 1
    #         question_template = random.choice(self.short_question_list)
    #         questions.append(question_template.format(class_name=text.lower()))
    #         answers.append(random.choice(self.answer_list))

    #     conversations = []
    #     conv = conversation_lib.default_conversation.copy()

    #     i = 0
    #     while i < len(questions):
    #         conv.messages = []
    #         conv.append_message(conv.roles[0], questions[i])
    #         conv.append_message(conv.roles[1], answers[i])
    #         conversations.append(conv.get_prompt())
    #         i += 1

    #     image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

    #     flag = False
    #     masks = []
    #     for ann_id in sampled_ann_ids:
    #         if isinstance(ann_id, list):
    #             flag = True
    #             if -1 in ann_id:
    #                 assert len(ann_id) == 1
    #                 m = np.zeros((image_info["height"], image_info["width"])).astype(
    #                     np.uint8
    #                 )
    #             else:
    #                 m_final = np.zeros(
    #                     (image_info["height"], image_info["width"])
    #                 ).astype(np.uint8)
    #                 for ann_id_i in ann_id:
    #                     ann = annotations[ann_id_i]

    #                     if len(ann["segmentation"]) == 0:
    #                         m = np.zeros(
    #                             (image_info["height"], image_info["width"])
    #                         ).astype(np.uint8)
    #                     else:
    #                         if type(ann["segmentation"][0]) == list:  # polygon
    #                             rle = mask.frPyObjects(
    #                                 ann["segmentation"],
    #                                 image_info["height"],
    #                                 image_info["width"],
    #                             )
    #                         else:
    #                             rle = ann["segmentation"]
    #                             for i in range(len(rle)):
    #                                 if not isinstance(rle[i]["counts"], bytes):
    #                                     rle[i]["counts"] = rle[i]["counts"].encode()
    #                         m = mask.decode(rle)
    #                         m = np.sum(
    #                             m, axis=2
    #                         )  # sometimes there are multiple binary map (corresponding to multiple segs)
    #                         m = m.astype(np.uint8)  # convert to np.uint8
    #                     m_final = m_final | m
    #                 m = m_final
    #             masks.append(m)
    #             continue

    #         ann = annotations[ann_id]

    #         if len(ann["segmentation"]) == 0:
    #             m = np.zeros((image_info["height"], image_info["width"])).astype(
    #                 np.uint8
    #             )
    #             masks.append(m)
    #             continue

    #         if type(ann["segmentation"][0]) == list:  # polygon
    #             rle = mask.frPyObjects(
    #                 ann["segmentation"], image_info["height"], image_info["width"]
    #             )
    #         else:
    #             rle = ann["segmentation"]
    #             for i in range(len(rle)):
    #                 if not isinstance(rle[i]["counts"], bytes):
    #                     rle[i]["counts"] = rle[i]["counts"].encode()
    #         m = mask.decode(rle)
    #         m = np.sum(
    #             m, axis=2
    #         )  # sometimes there are multiple binary map (corresponding to multiple segs)
    #         m = m.astype(np.uint8)  # convert to np.uint8
    #         masks.append(m)

    #     masks = np.stack(masks, axis=0)

    #     # if ds == 'grefcoco' and flag:
    #     #     import shutil
    #     #     image_name = image_path.split("/")[-1]
    #     #     save_dir = os.path.join("/group/30042/xlai/LISA_refactor_final/debug", image_name.split(".")[0])
    #     #     os.makedirs(save_dir, exist_ok=True)
    #     #     shutil.copy(image_path, save_dir)
    #     #     for i in range(masks.shape[0]):
    #     #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

    #     masks = torch.from_numpy(masks)
    #     label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

    #     return (
    #         image_path,
    #         image,
    #         image_clip,
    #         conversations,
    #         masks,
    #         label,
    #         resize,
    #         questions,
    #         sampled_classes,
    #     )
    def __getitem__(self, idx):
        ds = random.choice(self.refer_seg_ds_list)
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        image_info = random.choice(images)
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs.get(image_id, [])

        if len(refs) == 0:
            return self.__getitem__(0)

        # 문장 및 annotation id 수집
        sents, ann_ids = [], []
        for ref in refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"])
                ann_ids.append(ref["ann_id"])

        # 샘플링
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(len(sents), self.num_classes_per_sample, replace=False)
        else:
            sampled_inds = list(range(len(sents)))

        sampled_sents = [sents[i] for i in sampled_inds]
        sampled_ann_ids = [ann_ids[i] for i in sampled_inds]
        sampled_classes = sampled_sents  # keyword로 사용

        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # 질문/답변/대화 생성
        questions, answers = [], []
        for text in sampled_classes:
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for q, a in zip(questions, answers):
            conv.messages = []
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], a)
            conversations.append(conv.get_prompt())

        # 마스크 생성
        masks = []
        for ann_id in sampled_ann_ids:
            ann = annotations[ann_id]
            if not ann["segmentation"]:
                m = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            else:
                if isinstance(ann["segmentation"][0], list):  # polygon
                    rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                else:
                    rle = ann["segmentation"]
                    for r in rle:
                        if not isinstance(r["counts"], bytes):
                            r["counts"] = r["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(m, axis=2).astype(np.uint8)
            masks.append(m)
        masks = torch.from_numpy(np.stack(masks, axis=0))
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        keywords = sampled_classes

        return (
            image_path,         # 1
            image,              # 2
            image_clip,         # 3
            conversations,      # 4
            masks,              # 5
            label,              # 6
            resize,             # 7
            questions,          # 8
            sampled_classes,    # 9
            False,              # 10
            keywords,     # 11 (keyword로 사용)
        )

