#https://github.com/mc-lan/Text4Seg

import argparse
import torch
import os
from tqdm import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from model.segment_anything import SamPredictor, sam_model_registry
from model.segment_anything.utils.transforms import ResizeLongestSide

from model.llava.eval.Text4Seg_utils import compute_logits_from_mask, masks_sample_points, translate_sequence, decode_mask

from model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.conversation import conv_templates, SeparatorStyle
from model.llava.model.builder import load_pretrained_model
from model.llava.utils import disable_torch_init
from model.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from model.llava.eval.refer_seg_dataset import ValDataset
from model.llava.eval.question_answer_list import QUESTION_PARTIAL

from torch.utils.data import Dataset, DataLoader

import math
from model.KISA import LISAForCausalLM  # 사용자 정의 LISA 모델
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

def load_lisa_model(model_path, vision_tower="openai/clip-vit-large-patch14", precision="bf16", seg_token="[SEG]", max_length=512, load_in_4bit=False, load_in_8bit=False):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer(seg_token, add_special_tokens=False).input_ids[0]
    key_token_idx = tokenizer("[KEY]", add_special_tokens=False).input_ids[0]

    # Precision 설정
    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    # 로딩 설정
    kwargs = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"],
            ),
        })

    # 모델 로딩
    model = LISAForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        vision_tower=vision_tower,
        seg_token_idx=seg_token_idx,
        key_token_idx=key_token_idx,
        fusion_dim=256,
        proj_in_dim=4096,
        proj_out_dim=256,
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)

    vision_module = model.get_model().get_vision_tower()
    vision_module.to(dtype=torch_dtype)

    model = model.to(dtype=torch_dtype, device="cuda")
    model.eval()

    image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)

    return tokenizer, model, image_processor

# def get_chunk(ds, n, k):
#     chunk_size = math.ceil(len(ds) / n)  # integer division
#     i = chunk_size * k
#     ds.refer_seg_ds["images"] = ds.refer_seg_ds["images"][i:i + chunk_size]
#     return ds

def get_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)  # integer division
    i = chunk_size * k
    ds.refer_seg_ds["images"] = ds.refer_seg_ds["images"][i:i + chunk_size]
    return ds

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, sub_dataset, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dataset = sub_dataset

    def __getitem__(self, index):
        image, masks, questions, image_path = self.dataset[index]
        image_name = os.path.basename(image_path).split(".")[0]

        sample_list = []
        for question in questions:
            qs = random.choice(QUESTION_PARTIAL).replace("[class_name]", question.replace(",", ""))
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_new = image.resize((336, 336), Image.BILINEAR)
            image_clip = self.image_processor(images=image_new, return_tensors="pt")["pixel_values"][0]
            image_np = np.array(image_new)
            # image_pt = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()
            # image_pt = (image_pt - torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)) / torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
            # pad_h, pad_w = 1024 - image_pt.shape[1], 1024 - image_pt.shape[2]
            # image_pt = F.pad(image_pt, (0, pad_w, 0, pad_h)).unsqueeze(0)

            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            sample_list.append({
                "input_ids": input_id.squeeze(0),
                "image": image,
                "image_clip": image_clip,
                "resize": (336, 336),
                "original": image.size,
                "mask": masks,
                "image": image,
                "image_name": image_name,
                "question": question
            })

        return sample_list

    def __len__(self):
        return len(self.dataset)

    def __len__(self):
        return len(self.dataset)


# def collate_fn(batch):
#     input_ids, image_tensors, image_sizes, masks, image, image_name, questions, resize_list, original_size_list = zip(*batch)
#     return input_ids, image_tensors, image_sizes, masks, image, image_name, questions, resize_list, original_size_list

def collate_fn(batch):
    return batch[0]


# DataLoader
def create_data_loader(sub_dataset, tokenizer, image_processor, model_config, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(sub_dataset, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = os.path.basename(model_path.strip("/"))

    # LISA 모델 로딩
    tokenizer, model, image_processor = load_lisa_model(
        model_path=model_path,
        vision_tower=args.vision_tower if hasattr(args, "vision_tower") else "openai/clip-vit-large-patch14",
        precision=args.precision if hasattr(args, "precision") else "bf16",
        max_length=args.model_max_length if hasattr(args, "model_max_length") else 512,
        load_in_4bit=args.load_in_4bit if hasattr(args, "load_in_4bit") else False,
        load_in_8bit=args.load_in_8bit if hasattr(args, "load_in_8bit") else False
    )

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_path)
    sam = sam.to(dtype=torch.float32, device='cuda')
    predictor = SamPredictor(sam)

    val_dataset = ValDataset(args.image_folder, args.dataset_split)
    print(f"[INFO] Dataset size: {len(val_dataset)} samples.")  # 여기에 추가
    sub_dataset = get_chunk(val_dataset, args.num_chunks, args.chunk_idx)

    data_loader = create_data_loader(sub_dataset, tokenizer, image_processor, model.config)

    if "p16" in args.model_path:
        h, w = 16, 16
    elif "p24" in args.model_path:
        h, w = 24, 24
    else:
        h, w = 16, 16  # default

    for samples in tqdm(data_loader, total=len(data_loader)):
        for i, sample in enumerate(samples):
            input_ids = sample["input_ids"].unsqueeze(0).to(device='cuda')
            images_clip = sample["image_clip"].unsqueeze(0).to(dtype=torch.bfloat16, device='cuda')
            image = sample["image"]
            resize_list = [sample["resize"]]
            original_size_list = [sample["original"]]

            with torch.inference_mode():
                output_ids, pred_masks = model.evaluate(
                    images=images_clip,
                    # images=[image],
                    input_ids=input_ids,
                    resize_list=resize_list,
                    original_size_list=original_size_list,
                    tokenizer=tokenizer,
                )

            output_ids = output_ids[output_ids != IMAGE_TOKEN_INDEX]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            # 마스크 처리
            if len(pred_masks) == 0 or pred_masks[0].shape[0] == 0:
                pred_mask = torch.zeros((1, h, w), dtype=torch.uint8)
            else:
                pred_mask = pred_masks[0].detach().cpu()

            # Upsample to original size
            pred_mask_up = F.interpolate(pred_mask.unsqueeze(0).float(), size=sample["original"][::-1], mode='nearest').squeeze(0).squeeze(0)
            pred_mask_bin = (pred_mask_up > 0).long()

            if 1 not in pred_mask_bin:
                sam_mask = np.zeros((1, sample["original"][1], sample["original"][0]))
            else:
                logits = compute_logits_from_mask(pred_mask_bin)
                point_coords, point_labels = masks_sample_points(pred_mask_bin)

                sam_mask, score, logit = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=logits,
                    multimask_output=False
                )

                for _ in range(2):  # SAM refinement
                    sam_mask, score, logit = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=logit,
                        multimask_output=False
                    )

            gt_mask = sample["mask"][i]
            image = sample["image"]
            image_name = sample["image_name"]
            question = sample["question"]

            # Save paths
            ds_split = args.dataset_split.replace("|", "_")
            image_path = os.path.join(args.save_file, model_name, ds_split, image_name)
            os.makedirs(image_path, exist_ok=True)

            pred_mask_img = Image.fromarray(pred_mask_bin.cpu().numpy().astype("uint8") * 255).convert("L")
            sam_mask_img = Image.fromarray(sam_mask[0].astype("uint8") * 255).convert("L")
            gt_mask_img = Image.fromarray(gt_mask.cpu().numpy().astype("uint8") * 255).convert("L")

            pred_mask_img.save(os.path.join(image_path, f"{i}_pred_mask.png"))
            sam_mask_img.save(os.path.join(image_path, f"{i}_sam_mask.png"))
            gt_mask_img.save(os.path.join(image_path, f"{i}_gt_mask.png"))
            image.save(os.path.join(image_path, f"{i}_image.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/hrkim/dataset/refer_seg/")
    parser.add_argument("--sam_path", type=str, default="/home/hrkim/dataset/sam_vit_h_4b8939.pth")
    parser.add_argument("--dataset_split", type=str, default="refcoco|unc|val")
    parser.add_argument("--save_file", type=str, default="./eval/ref_seg_results/")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3069)
    
    # 추가된 LISA 모델 관련 인자들
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    
    args = parser.parse_args()

    eval_model(args)