import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

# from model.LISA import LISAForCausalLM
from model.KISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="runs/lisa_finetune_key_fusion_ce_es")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--ckpt", default="", type=str, help="(Unused) Path to checkpoint")
    parser.add_argument("--fusion_dim", default=256)
    parser.add_argument("--proj_in_dim", default=4096)
    parser.add_argument("--proj_out_dim", default=256)
    return parser.parse_args(args)

def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024):
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, "[SEG]", "[KEY]"])
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.key_token_idx = tokenizer("[KEY]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16": torch_dtype = torch.bfloat16
    elif args.precision == "fp16": torch_dtype = torch.half

    model = LISAForCausalLM.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        seg_token_idx=args.seg_token_idx,
        key_token_idx=args.key_token_idx,
        vision_tower=args.vision_tower,
        fusion_dim=args.fusion_dim,
        proj_in_dim=args.proj_in_dim,
        proj_out_dim=args.proj_out_dim
    )
    model.resize_token_embeddings(len(tokenizer))

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower().to(dtype=torch_dtype, device=args.local_rank)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and not args.load_in_4bit and not args.load_in_8bit:
        model = model.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()
    
    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = clip_image_processor(images=image_np, return_tensors="pt")["pixel_values"].to(dtype=torch_dtype, device=args.local_rank)

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(dtype=torch_dtype, device=args.local_rank)

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device=args.local_rank)

        input_dict = {
            "images": image,
            "images_clip": image_clip,
            "input_ids": input_ids,
            "resize_list": resize_list,
            "original_size_list": original_size_list,
            "max_new_tokens": 512,
            "tokenizer": tokenizer
        }

        # with torch.no_grad():
        #     output_dict = model(**input_dict)

        # output_ids = output_dict["output_ids"][0]
        # pred_masks = output_dict["pred_masks"]
        output_ids, pred_masks = model.evaluate(
            images_clip=image_clip,
            images=image,
            input_ids=input_ids,
            resize_list=resize_list,
            original_size_list=original_size_list,
            tokenizer=tokenizer,
        )

        output_ids = output_ids[output_ids != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False).replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue
            pred_mask = pred_mask.detach().cpu().numpy()[0] > 0

            save_path = f"{args.vis_save_path}/{os.path.basename(image_path).split('.')[0]}_mask_{i}.jpg"
            cv2.imwrite(save_path, pred_mask.astype(np.uint8) * 255)
            print(f"{save_path} has been saved.")

            save_img = image_np.copy()
            save_img[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None] * np.array([255, 0, 0]) * 0.5)[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            masked_path = f"{args.vis_save_path}/{os.path.basename(image_path).split('.')[0]}_masked_img_{i}.jpg"
            cv2.imwrite(masked_path, save_img)
            print(f"{masked_path} has been saved.")

if __name__ == "__main__":
    main(sys.argv[1:])
