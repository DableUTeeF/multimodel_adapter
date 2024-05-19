import torch.nn as nn
import clip
from transformers import AutoModelForCausalLM
from timm.models.vision_transformer import Block


class HF_adapter(nn.Module):
    def __init__(self, model_name,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 w_bias=False,
                 w_lora=False, lora_rank=16,
                 w_new_gate=False,
                 phase="finetune"
                 ):
        w_bias = phase == "finetune"

        self.clip, self.clip_transform = clip.load(clip_model)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, self.model.config.hidden_size)
        self.visual_proj_norm = nn.LayerNorm(self.model.config.hidden_size)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, self.model.config.hidden_size)

        del self.clip.transformer

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.phase = phase
        self.get_trainable_params(self.phase)

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("model."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        else:
            raise ValueError(f"Unknown model phase: {phase}")

