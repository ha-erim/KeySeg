from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

class LightSelfAttention(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # self.reset_parameters()

        # 🔵 out_proj는 float32로 강제 유지
        self.out_proj = self.out_proj.to(torch.float32)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, N, C = x.shape

        x = torch.clamp(x, min=-1e4, max=1e4)  # 🔵 입력 클램프

        # 🔵 Projection 후 float32 변환
        q = self.q_proj(x).to(torch.float32).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).to(torch.float32).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).to(torch.float32).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = out.to(torch.bfloat16) 

        out = self.out_proj(out)  # 🔵 여기서 out_proj는 float32 weight

        return out  # float32 출력 (최종 bf16 변환은 KeywordSegFusionModule에서)


class KeywordSegFusionModule(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=256, upsample_size=12):
        super().__init__()
        self.query_proj = nn.Linear(vision_dim, hidden_dim)
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vision_dim)

        # 🔵 out_proj는 float32로 강제 유지
        self.out_proj = self.out_proj.to(torch.float32)

        self.self_attn = LightSelfAttention(dim=vision_dim)
        self.norm1 = nn.LayerNorm(vision_dim)
        self.norm2 = nn.LayerNorm(vision_dim)
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 2),
            nn.GELU(),
            nn.Linear(vision_dim * 2, vision_dim),
        )

        self.upsample_size = upsample_size

    def forward(self, key_list, segs_list):
        fused_segs = []
        for i in range(len(segs_list)):
            seg = segs_list[i]
            key = key_list[i]

            if key.dim() == 1:
                key = key.unsqueeze(0)

            seg = seg.unsqueeze(0).transpose(1, 2)
            upsampled_seg = F.interpolate(seg, size=self.upsample_size, mode='linear', align_corners=False)
            upsampled_seg = upsampled_seg.transpose(1, 2).squeeze(0)

            Q = self.query_proj(upsampled_seg).to(torch.float32)
            K = self.key_proj(key).to(torch.float32)
            V = self.value_proj(key).to(torch.float32)

            attn_scores = torch.matmul(Q, K.transpose(0, 1)) / (Q.size(-1) ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, V)

            context = context.to(torch.bfloat16)  # 🔥 context를 bf16로 변환

            context = self.out_proj(context)

            seg = upsampled_seg + context
            # seg = torch.nan_to_num(seg, nan=0.0, posinf=1e3, neginf=-1e3)
            # seg = self.norm1(seg)

            seg_sa = self.self_attn(seg.unsqueeze(0)).squeeze(0)
            # seg = self.norm2(seg + seg_sa)

            seg = seg + self.mlp(seg)

            seg = seg.unsqueeze(0).transpose(1, 2)
            seg = F.interpolate(seg, size=segs_list[i].size(0), mode='linear', align_corners=False)
            seg = seg.transpose(1, 2).squeeze(0)

            seg = seg.to(torch.bfloat16)  # 최종 bf16으로 통일

            fused_segs.append(seg)
        return fused_segs



def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # if not hasattr(config, "train_mask_decoder"):       
        if not config.train_mask_decoder:
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.key_loss_weight = kwargs.pop("key_loss_weight",None)
        
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.key_token_idx = kwargs.pop("key_token_idx")
        self.fusion_dim = kwargs.pop("fusion_dim")
        self.proj_in_dim = kwargs.pop("proj_in_dim")
        self.proj_out_dim = kwargs.pop("proj_out_dim")
        self.temperature = config.temperature
        
        super().__init__(config)
        
        self.keyword_fusion = KeywordSegFusionModule(
            text_dim=self.fusion_dim, #256
            vision_dim=self.fusion_dim,
        )

        self.keyword_text_proj = nn.Linear(
           self.proj_in_dim, self.proj_out_dim # 4096,256
        )

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embedding_table_proj = nn.Linear(4096, 256)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        input_keywords_ids_list: List[str],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat([seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()], dim=1)
        seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask], dim=1)

        key_token_mask = input_ids[:, 1:] == self.key_token_idx
        key_token_mask = torch.cat([key_token_mask, torch.zeros((key_token_mask.shape[0], 1)).bool().cuda()], dim=1)
        key_token_mask = torch.cat([torch.zeros((key_token_mask.shape[0], 255)).bool().cuda(), key_token_mask], dim=1)

        if inference:
            length = input_ids.shape[0]
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output = super().forward(
                images=images_clip_extend,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
        else:
            images_clip = torch.cat([
                images_clip[i].unsqueeze(0).expand(offset[i+1] - offset[i], -1, -1, -1)
                for i in range(len(offset) - 1)
            ], dim=0)
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = self.model.text_hidden_fcs[0](output_hidden_states[-1])
        last_hidden_state = hidden_states
        
        if inference:
            last_hidden_state = last_hidden_state.view(input_ids.shape[0], -1, last_hidden_state.shape[-1])  # (B, T, D)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_mask.int().sum(-1).cumsum(-1)], dim=0
        )[offset]
        pred_embeddings = [pred_embeddings[seg_token_offset[i]:seg_token_offset[i+1]] for i in range(batch_size)]

        key_embeddings = last_hidden_state[key_token_mask]
        key_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), key_token_mask.int().sum(-1).cumsum(-1)], dim=0
        )[offset]
        key_embedding = [key_embeddings[key_token_offset[i]:key_token_offset[i+1]] for i in range(batch_size)]

        if not hasattr(self, "keyword_fusion"):
            self.keyword_fusion = KeywordSegFusionModule(
                text_dim=key_embedding[0].shape[-1],
                vision_dim=pred_embeddings[0].shape[-1]
            ).to(pred_embeddings[0].device)

        pred_embeddings = self.keyword_fusion(key_embedding, pred_embeddings)

        pred_masks = []
        for i in range(batch_size):
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        if inference:
            return {"pred_masks": pred_masks, "gt_masks": masks_list}

        # Keyword Loss (embedding → id → loss)
        keyword_loss = 0
        if len(key_embedding) > 0:
            embedding_table = self.model.get_input_embeddings().weight  # (V, 4096)
            embedding_table_proj = self.embedding_table_proj(embedding_table)  # (V, 256)

            # normalize
            key_embedding_flattened = torch.cat(key_embedding, dim=0)
            key_embedding_norm = F.normalize(key_embedding_flattened, dim=-1)
            table_norm = F.normalize(embedding_table_proj, dim=-1)

            # temperature scaling
            logits = torch.matmul(key_embedding_norm, table_norm.T) / self.temperature  # (N, V)

            # target
            target_ids = [
                ids[0] if len(ids) > 0 else self.model.config.pad_token_id
                for ids in input_keywords_ids_list
            ]
            target_ids = torch.tensor(target_ids, dtype=torch.long, device=logits.device)

            loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
            keyword_loss = loss_fn(logits, target_ids) * self.key_loss_weight



        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for pred_mask, gt_mask in zip(pred_masks, masks_list):
            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss /= (num_masks + 1e-8)
        mask_dice_loss /= (num_masks + 1e-8)
        mask_loss = self.bce_loss_weight * mask_bce_loss + self.dice_loss_weight * mask_dice_loss

        return {
            "loss": ce_loss + mask_loss + keyword_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "keyword_loss": keyword_loss,
        }


    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=32,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
                dim=1,
            )
            
            key_token_mask = output_ids[:, 1:] == self.key_token_idx
            key_token_mask = torch.cat(
                [torch.zeros((key_token_mask.shape[0], 255)).bool().cuda(), key_token_mask],
                dim=1,
            )

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            # === Extract embeddings ===
            seg_embeddings = last_hidden_state[seg_token_mask]
            key_embeddings = last_hidden_state[key_token_mask]

            # === Offset 계산 ===
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            key_token_indices = torch.nonzero(key_token_mask, as_tuple=False)
            keyword_embeds = []
            for i in range(images.shape[0]):
                indices = key_token_indices[key_token_indices[:, 0] == i]
                if len(indices) > 0:
                    keyword_embeds.append(key_embeddings[indices[0][0]])
                else:
                    keyword_embeds.append(torch.zeros_like(seg_embeddings[0]))
            keyword_embeds = torch.stack(keyword_embeds, dim=0)

            # === seg embedding을 각 sample별로 분할 ===
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(seg_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            # === keyword fusion ===
            pred_embeddings = self.keyword_fusion(keyword_embeds, pred_embeddings)

            # === Mask prediction ===
            image_embeddings = self.get_visual_embs(images)
            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks