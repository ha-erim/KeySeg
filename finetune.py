import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
    
from model.KISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.reason_seg_key_dataset import ValDataset, collate_fn_val
from utils.reason_refer import HybridDataset, collate_fn
from utils.refer_seg_dataset import ReferSegValDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

import matplotlib.pyplot as plt
import uuid
from datetime import datetime

def visualize_masks(image_tensor, gt_mask, pred_mask, save_path, idx=0, epoch=0,
                    prompt: str = "", image_path: str = ""):
    # 저장 경로에 epoch 디렉토리 추가
    epoch_dir = os.path.join(save_path, f"epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # === 정규화 되지 않은 이미지로 복원 ===
    image_np = image_tensor[idx].cpu().numpy()
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)  # CHW → HWC

    if image_np.max() <= 1.0:
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    pred_np = (pred_mask[idx].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    gt_np = (gt_mask[idx].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # === 시각화 저장 ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_np, cmap='gray')
    axs[1].set_title("GT Mask")
    axs[1].axis('off')

    axs[2].imshow(pred_np, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    # 텍스트로 프롬프트 및 이미지 경로 출력
    fig.suptitle(f"Prompt: {prompt}\nImage Path: {image_path}", fontsize=10, y=1.05)

    plt.tight_layout()

    # 순차 저장: sample_1.png, sample_2.png ...
    existing = [f for f in os.listdir(epoch_dir) if f.endswith(".png")]
    sample_idx = len(existing) + 1
    save_file = os.path.join(epoch_dir, f"sample_{sample_idx}.png")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    
def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model finetuning")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="xinlai/LISA-7B-v1")
    parser.add_argument("--vis_save_path", default="./mask_vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="refer_seg||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3", type=str)
    # parser.add_argument(
    #     "--sem_seg_data",
    #     default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
    #     type=str,
    # )
    parser.add_argument(
        "--refer_seg_data", default="refcoco||refcoco+||refcocog", type=str
    )
    # parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--test_dataset", default="refcocog|umd|test", type=str)
    # parser.add_argument("--test_dataset", default="ReasonSeg|test", type=str)
    parser.add_argument("--dataset_dir", default="/home/hrkim/dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--steps_per_epoch", default=200, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1e-4, type=float) #1.0
    parser.add_argument("--dice_loss_weight", default=0.5, type=float) #2.0
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--key_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=1.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    
    parser.add_argument("--test_mode", action="store_true")
    # parser.add_argument("--ckpt", default="/home/hrkim/LISA/runs/lisa_finetune_key/meta_log_giou0.465_ciou0.492.pth", type=str)
    parser.add_argument("--ckpt", default="")
    
    parser.add_argument("--vision_pretrained", default="/home/hrkim/dataset/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=30, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    # parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    
    parser.add_argument("--fusion_dim", default=256)
    parser.add_argument("--proj_in_dim", default=4096)
    parser.add_argument("--proj_out_dim", default=256)
    parser.add_argument("--temperature", default=0.07)
    
    return parser.parse_args(args)

def revise_state_dict_keys(state_dict, prefix_to_remove="base_model.model."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix_to_remove):
            new_k = k[len(prefix_to_remove):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    num_added_seg_tokens = tokenizer.add_tokens("[SEG]")
    num_added_key_tokens = tokenizer.add_tokens("[KEY]")    

    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.key_token_idx = tokenizer("[KEY]",add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "key_loss_weight":args.key_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "key_token_idx": args.key_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "fusion_dim":args.fusion_dim,
        "proj_in_dim":args.proj_in_dim,
        "proj_out_dim":args.proj_out_dim,
        "temperature":args.temperature,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
            
    # === 🔍 학습 가능 파라미터 확인 ===
    print("Trainable parameters with non-zero norm:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: norm={param.data.norm().item():.4f}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    # train_dataset = ReasonSegKeyDataset(
    #     args.dataset_dir,
    #     tokenizer,
    #     args.vision_tower,
    #     samples_per_epoch=args.batch_size,
    #     precision=args.precision,
    #     image_size=args.image_size,
    #     num_classes_per_sample=args.num_classes_per_sample,
    #     exclude_val=args.exclude_val,
    #     reason_seg_data=args.reason_seg_data,
    #     explanatory=args.explanatory,
    # )
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        refer_seg_data=args.refer_seg_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
    )
    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")
    
    if args.test_mode:
        # test_dataset = ValDataset(
        #     args.dataset_dir,
        #     tokenizer,
        #     args.vision_tower,
        #     args.test_dataset,
        #     args.image_size,
        # )
        test_dataset = ReferSegValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.test_dataset,
            args.image_size,
        )
        print(f"Testing with {len(test_dataset)} examples.")
    else:
        test_dataset = None
        print(f"Test_dataset is None")
        
        
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    
    #resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume
    
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

        
    if args.test_mode:
        if args.local_rank == 0:
            print(f"[Evaluation Mode] Loading checkpoint weights from: {args.ckpt}")
        print("Loading state_dict from ckpt")
        load_path, client_state = model_engine.load_checkpoint("runs/key_fusion_ce_es_upsamle_sa_tmd_ref_200/ckpt_model")



    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_val,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    
    # Test dataset
    if test_dataset is not None:
        assert args.val_batch_size == 1
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=test_sampler,
            collate_fn=partial(
                collate_fn_val,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0
    
    # === Early stopping 설정 ===
    patience = 5
    counter = 0

    if args.eval_only:
        # ReasonSeg|test
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()
    
    if args.test_mode == True:
        # ReasonSeg|test
        print("evaluation with testset")
        giou, ciou = validate(test_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if not args.no_eval:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score

            if is_best:
                best_score = giou
                cur_ciou = ciou
                counter = 0  # 성능이 향상되면 카운터 초기화
            else:
                counter += 1
                print(f"[EarlyStopping] Epoch {epoch}: No improvement. Patience {counter}/{patience}")

            if counter >= patience:
                print(f"[EarlyStopping] Stopping early at epoch {epoch}.")
                break

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    keyword_losses = AverageMeter("KeywordLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            keyword_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for global_step in tqdm.trange(args.steps_per_epoch, desc=f"Epoch {epoch}"):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            keyword_loss = output_dict["keyword_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            keyword_losses.update(keyword_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        total_step = epoch * args.steps_per_epoch + global_step

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                keyword_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, total_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, total_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, total_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, total_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, total_step)
                writer.add_scalar("train/keyword_loss", keyword_losses.avg, total_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, total_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, total_step)
                writer.flush()

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            keyword_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], total_step)


    return train_iter



def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1
        
        # === 시각화 추가 ===
        if args.local_rank == 0:
            visualize_masks(
                image_tensor=input_dict["images"].float().cpu(),
                gt_mask=masks_list,
                pred_mask=(pred_masks[0] > 0).float().cpu(),
                save_path=args.vis_save_path,
                idx=0,
                epoch=epoch,
            )

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        writer.flush()

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
