import os
import torch
import argparse
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
# from model.LISA import LISAForCausalLM
from model.KISA import LISAForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to Hugging Face format")
    parser.add_argument("--version", type=str, default="xinlai/LISA-7B-v1")
    parser.add_argument("--deepspeed_ckpt", type=str,
                        default="runs/key_fusion_ce_es_upsamle_sa_tmd_ref_200/ckpt_model/global_step600/mp_rank_00_model_states.pt")
    parser.add_argument("--vision_pretrained", type=str, default="/home/hrkim/dataset/sam_vit_h_4b8939.pth")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    return parser.parse_args()


def main():
    args = parse_args()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.version, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["[SEG]", "[KEY]", DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    key_token_idx = tokenizer("[KEY]", add_special_tokens=False).input_ids[0]

    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": 1e-4,
        "dice_loss_weight": 2.0,
        "bce_loss_weight": 2.0,
        "key_loss_weight": 0.5,
        "seg_token_idx": seg_token_idx,
        "key_token_idx": key_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": "openai/clip-vit-large-patch14",
        "use_mm_start_end": True,
        "fusion_dim": 256,
        "proj_in_dim": 4096,
        "proj_out_dim": 256,
    }

    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.resize_token_embeddings(len(tokenizer))

    # === Load DeepSpeed checkpoint ===
    print(f"🔄 Loading DeepSpeed checkpoint from: {args.deepspeed_ckpt}")
    ckpt = torch.load(args.deepspeed_ckpt, map_location="cpu")
    print(ckpt)
    state_dict = ckpt.get("module", ckpt)  # for DeepSpeed, model weights are under "module"
    model.load_state_dict(state_dict, strict=False)

    # === Save as Hugging Face format ===
    save_path = "runs/key_fusion_ce_es_upsamle_sa_tmd_ref_200/hf"
    print(f"Saving merged model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Done")


if __name__ == "__main__":
    main()