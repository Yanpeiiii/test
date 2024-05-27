import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import copy
import math
import numpy as np


class ChannelTransformerEmbed(nn.Module):
    def __init__(self, opt, img_size, patch_size, stride_size, in_channels, channel_num):
        super(ChannelTransformerEmbed, self).__init__()
        img_size = _pair(img_size)
        num_patches = (img_size[0] // stride_size) * (img_size[1] // stride_size)
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=channel_num,
                                          kernel_size=patch_size, stride=stride_size,
                                          padding=(patch_size // 2, patch_size // 2))

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, in_channels))
        self.dropout = nn.Dropout(opt.channel_embeddings_dropout_rate)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention1(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(Attention1, self).__init__()
        self.vis = vis
        self.KV_size = channel_num[0] + channel_num[1]
        self.channel_num = channel_num
        self.num_attention_heads = opt.channel_transformer_num_heads

        self.query = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):
            query = nn.Linear(channel_num[0], channel_num[0], bias=False)
            key = nn.Linear(self.KV_size, self.KV_size, bias=False)
            value = nn.Linear(self.KV_size, self.KV_size, bias=False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)
        self.out = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.attn_dropout = nn.Dropout(opt.channel_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(opt.channel_attention_dropout_rate)

    def forward(self, emb, emb_all):
        multi_head_q_list = []
        multi_head_k_list = []
        multi_head_v_list = []
        if emb is not None:
            for query in self.query:
                q = query(emb)
                multi_head_q_list.append(q)

        for key in self.key:
            k = key(emb_all)
            multi_head_k_list.append(k)
        for value in self.value:
            v = value(emb_all)
            multi_head_v_list.append(v)

        multi_head_q = torch.stack(multi_head_q_list, dim=1)
        multi_head_k = torch.stack(multi_head_k_list, dim=1)
        multi_head_v = torch.stack(multi_head_v_list, dim=1)

        multi_head_q = multi_head_q.transpose(-1, -2)

        attention_scores = torch.matmul(multi_head_q, multi_head_k)
        attention_scores = attention_scores / math.sqrt(self.KV_size)
        attention_probs = self.softmax(self.psi(attention_scores))

        if self.vis:
            weights = []
            weights.append(attention_probs.mean(1))
        else:
            weights = None

        attention_probs = self.attn_dropout(attention_probs)

        multi_head_v = multi_head_v.transpose(-1, -2)
        context_layer = torch.matmul(attention_probs, multi_head_v)
        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        context_layer = context_layer.mean(dim=3)
        o = self.out(context_layer)
        o = self.proj_dropout(o)

        return o, weights


class Attention2(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(Attention2, self).__init__()
        self.vis = vis
        self.KV_size = channel_num[0] + channel_num[1] + channel_num[2]
        self.channel_num = channel_num
        self.num_attention_heads = opt.channel_transformer_num_heads

        self.query = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):
            query = nn.Linear(channel_num[1], channel_num[1], bias=False)
            key = nn.Linear(self.KV_size, self.KV_size, bias=False)
            value = nn.Linear(self.KV_size, self.KV_size, bias=False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)
        self.out = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.attn_dropout = nn.Dropout(opt.channel_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(opt.channel_attention_dropout_rate)

    def forward(self, emb, emb_all):
        multi_head_q_list = []
        multi_head_k_list = []
        multi_head_v_list = []
        if emb is not None:
            for query in self.query:
                q = query(emb)
                multi_head_q_list.append(q)

        for key in self.key:
            k = key(emb_all)
            multi_head_k_list.append(k)
        for value in self.value:
            v = value(emb_all)
            multi_head_v_list.append(v)

        multi_head_q = torch.stack(multi_head_q_list, dim=1)
        multi_head_k = torch.stack(multi_head_k_list, dim=1)
        multi_head_v = torch.stack(multi_head_v_list, dim=1)

        multi_head_q = multi_head_q.transpose(-1, -2)

        attention_scores = torch.matmul(multi_head_q, multi_head_k)
        attention_scores = attention_scores / math.sqrt(self.KV_size)
        attention_probs = self.softmax(self.psi(attention_scores))

        if self.vis:
            weights = []
            weights.append(attention_probs.mean(1))
        else:
            weights = None

        attention_probs = self.attn_dropout(attention_probs)
        multi_head_v = multi_head_v.transpose(-1, -2)
        context_layer = torch.matmul(attention_probs, multi_head_v)
        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        context_layer = context_layer.mean(dim=3)
        o = self.out(context_layer)
        o = self.proj_dropout(o)
        return o, weights


class Attention3(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(Attention3, self).__init__()
        self.vis = vis
        self.KV_size = channel_num[0] + channel_num[1] + channel_num[2]
        self.channel_num = channel_num
        self.num_attention_heads = opt.channel_transformer_num_heads

        self.query = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):
            query = nn.Linear(channel_num[1], channel_num[1], bias=False)
            key = nn.Linear(self.KV_size, self.KV_size, bias=False)
            value = nn.Linear(self.KV_size, self.KV_size, bias=False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)
        self.out = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.attn_dropout = nn.Dropout(opt.channel_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(opt.channel_attention_dropout_rate)

    def forward(self, emb, emb_all):
        multi_head_q_list = []
        multi_head_k_list = []
        multi_head_v_list = []
        if emb is not None:
            for query in self.query:
                q = query(emb)
                multi_head_q_list.append(q)

        for key in self.key:
            k = key(emb_all)
            multi_head_k_list.append(k)
        for value in self.value:
            v = value(emb_all)
            multi_head_v_list.append(v)

        multi_head_q = torch.stack(multi_head_q_list, dim=1)
        multi_head_k = torch.stack(multi_head_k_list, dim=1)
        multi_head_v = torch.stack(multi_head_v_list, dim=1)

        multi_head_q = multi_head_q.transpose(-1, -2)

        attention_scores = torch.matmul(multi_head_q, multi_head_k)
        attention_scores = attention_scores / math.sqrt(self.KV_size)
        attention_probs = self.softmax(self.psi(attention_scores))

        if self.vis:
            weights = []
            weights.append(attention_probs.mean(1))
        else:
            weights = None

        attention_probs = self.attn_dropout(attention_probs)

        multi_head_v = multi_head_v.transpose(-1, -2)
        context_layer = torch.matmul(attention_probs, multi_head_v)
        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        context_layer = context_layer.mean(dim=3)
        o = self.out(context_layer)
        o = self.proj_dropout(o)

        return o, weights


class Attention4(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(Attention4, self).__init__()
        self.vis = vis
        self.KV_size = channel_num[0] + channel_num[1]
        self.channel_num = channel_num
        self.num_attention_heads = opt.channel_transformer_num_heads

        self.query = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):
            query = nn.Linear(channel_num[1], channel_num[1], bias=False)
            key = nn.Linear(self.KV_size, self.KV_size, bias=False)
            value = nn.Linear(self.KV_size, self.KV_size, bias=False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)
        self.out = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.attn_dropout = nn.Dropout(opt.channel_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(opt.channel_attention_dropout_rate)

    def forward(self, emb, emb_all):
        multi_head_q_list = []
        multi_head_k_list = []
        multi_head_v_list = []
        if emb is not None:
            for query in self.query:
                q = query(emb)
                multi_head_q_list.append(q)

        for key in self.key:
            k = key(emb_all)
            multi_head_k_list.append(k)
        for value in self.value:
            v = value(emb_all)
            multi_head_v_list.append(v)

        multi_head_q = torch.stack(multi_head_q_list, dim=1)
        multi_head_k = torch.stack(multi_head_k_list, dim=1)
        multi_head_v = torch.stack(multi_head_v_list, dim=1)

        multi_head_q = multi_head_q.transpose(-1, -2)

        attention_scores = torch.matmul(multi_head_q, multi_head_k)
        attention_scores = attention_scores / math.sqrt(self.KV_size)
        attention_probs = self.softmax(self.psi(attention_scores))

        if self.vis:
            weights = []
            weights.append(attention_probs.mean(1))
        else:
            weights = None

        attention_probs = self.attn_dropout(attention_probs)

        multi_head_v = multi_head_v.transpose(-1, -2)
        context_layer = torch.matmul(attention_probs, multi_head_v)
        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        context_layer = context_layer.mean(dim=3)
        o = self.out(context_layer)
        o = self.proj_dropout(o)

        return o, weights


class ChannelTransformerModuleMlp(nn.Module):
    def __init__(self, opt, in_channel, mlp_channel):
        super(ChannelTransformerModuleMlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(opt.channel_mlp_dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BlockViTStage1(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(BlockViTStage1, self).__init__()
        expand_ratio = opt.channel_transformer_expand_ratio
        self.attn_norm1 = nn.LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm = nn.LayerNorm(channel_num[0]+channel_num[1], eps=1e-6)
        self.channel_attn = Attention1(opt, vis, channel_num)

        self.ffn_norm1 = nn.LayerNorm(channel_num[0], eps=1e-6)
        self.ffn1 = ChannelTransformerModuleMlp(opt, channel_num[0], channel_num[0]*expand_ratio)

    def forward(self, emb1, emb2):
        embcat = []
        org1 = emb1
        for i in range(2):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        cx1 = self.attn_norm1(emb1)
        emb_all = self.attn_norm(emb_all)
        cx1, weights = self.channel_attn(cx1, emb_all)
        cx1 = org1 + cx1

        org1 = cx1
        x1 = self.ffn_norm1(cx1)
        x1 = self.ffn1(x1)
        x1 = x1 + org1

        return x1


class BlockViTStage2(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(BlockViTStage2, self).__init__()
        expand_ratio = opt.channel_transformer_expand_ratio
        self.attn_norm1 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm = nn.LayerNorm(channel_num[0]+channel_num[1]+channel_num[2], eps=1e-6)
        self.channel_attn = Attention2(opt, vis, channel_num)

        self.ffn_norm1 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.ffn1 = ChannelTransformerModuleMlp(opt, channel_num[1], channel_num[1]*expand_ratio)

    def forward(self, emb1, emb2, emb3):
        embcat = []
        org2 = emb2
        for i in range(3):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        cx2 = self.attn_norm1(emb2)
        emb_all = self.attn_norm(emb_all)
        cx2, weights = self.channel_attn(cx2, emb_all)
        cx2 = org2 + cx2

        org2 = cx2
        x2 = self.ffn_norm1(cx2)
        x2 = self.ffn1(x2)
        x2 = x2 + org2

        return x2


class BlockViTStage3(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(BlockViTStage3, self).__init__()
        expand_ratio = opt.channel_transformer_expand_ratio
        self.attn_norm1 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm = nn.LayerNorm(channel_num[0]+channel_num[1]+channel_num[2], eps=1e-6)
        self.channel_attn = Attention3(opt, vis, channel_num)

        self.ffn_norm1 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.ffn1 = ChannelTransformerModuleMlp(opt, channel_num[1], channel_num[1]*expand_ratio)

    def forward(self, emb1, emb2, emb3):
        embcat = []
        org2 = emb2
        for i in range(3):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        cx2 = self.attn_norm1(emb2)
        emb_all = self.attn_norm(emb_all)
        cx2, weights = self.channel_attn(cx2, emb_all)
        cx2 = org2 + cx2

        org2 = cx2
        x2 = self.ffn_norm1(cx2)
        x2 = self.ffn1(x2)
        x2 = x2 + org2

        return x2


class BlockViTStage4(nn.Module):
    def __init__(self, opt, vis, channel_num):
        super(BlockViTStage4, self).__init__()
        expand_ratio = opt.channel_transformer_expand_ratio
        self.attn_norm1 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm = nn.LayerNorm(channel_num[0]+channel_num[1], eps=1e-6)
        self.channel_attn = Attention4(opt, vis, channel_num)

        self.ffn_norm1 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.ffn1 = ChannelTransformerModuleMlp(opt, channel_num[1], channel_num[1]*expand_ratio)

    def forward(self, emb1, emb2):
        embcat = []
        org2 = emb2
        for i in range(2):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        cx2 = self.attn_norm1(emb2)
        emb_all = self.attn_norm(emb_all)
        cx2, weights = self.channel_attn(cx2, emb_all)
        cx2 = org2 + cx2

        org2 = cx2
        x1 = self.ffn_norm1(cx2)
        x1 = self.ffn1(x1)
        x1 = x1 + org2

        return x1


class Reconstruct1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct1, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=self.scale_factor,
                                            stride=self.scale_factor)
        self.upsample2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=self.scale_factor,
                                            stride=self.scale_factor)

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.upsample1(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.upsample2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)

        return out


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=self.scale_factor,
                                            stride=self.scale_factor)

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.upsample1(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)

        return out


class ChannelTransformerStage(nn.Module):
    def __init__(self, opt, vis=None, img_size=256, in_channels=(64, 128, 256, 512), channel_num=(64, 128, 256, 512),
                 patch_size=(7, 3), strize_size=(4, 2, 1)):
        super(ChannelTransformerStage, self).__init__()

        self.vis = vis
        self.patchSize_7 = patch_size[0]
        self.patchSize_3 = patch_size[1]
        self.strizeSize_4 = strize_size[0]
        self.strizeSize_2 = strize_size[1]
        self.strizeSize_1 = strize_size[2]

        self.embeddings_stage_1_1 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_7, img_size=img_size,
                                                            stride_size=self.strizeSize_4, in_channels=in_channels[0],
                                                            channel_num=channel_num[0])
        self.embeddings_stage_1_2 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_3, img_size=img_size // 2,
                                                            stride_size=self.strizeSize_2, in_channels=in_channels[1],
                                                            channel_num=channel_num[1])

        self.embeddings_stage_2_1 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_7, img_size=img_size,
                                                            stride_size=self.strizeSize_4, in_channels=in_channels[0],
                                                            channel_num=channel_num[0])
        self.embeddings_stage_2_2 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_3, img_size=img_size // 2,
                                                            stride_size=self.strizeSize_2, in_channels=in_channels[1],
                                                            channel_num=channel_num[1])
        self.embeddings_stage_2_3 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_3, img_size=img_size // 4,
                                                            stride_size=self.strizeSize_1, in_channels=in_channels[2],
                                                            channel_num=channel_num[2])

        self.embeddings_stage_3_1 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_7, img_size=img_size // 2,
                                                            stride_size=self.strizeSize_4, in_channels=in_channels[1],
                                                            channel_num=channel_num[1])
        self.embeddings_stage_3_2 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_3, img_size=img_size // 4,
                                                            stride_size=self.strizeSize_2, in_channels=in_channels[2],
                                                            channel_num=channel_num[2])
        self.embeddings_stage_3_3 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_3, img_size=img_size // 8,
                                                            stride_size=self.strizeSize_1, in_channels=in_channels[3],
                                                            channel_num=channel_num[3])

        self.embeddings_stage_4_1 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_7, img_size=img_size // 4,
                                                            stride_size=self.strizeSize_4, in_channels=in_channels[2],
                                                            channel_num=channel_num[2])
        self.embeddings_stage_4_2 = ChannelTransformerEmbed(opt, patch_size=self.patchSize_3, img_size=img_size // 8,
                                                            stride_size=self.strizeSize_2, in_channels=in_channels[3],
                                                            channel_num=channel_num[3])

        self.attention_1_1 = BlockViTStage1(opt, vis, channel_num=[channel_num[0], channel_num[1]])
        self.attention_1_2 = BlockViTStage2(opt, vis, channel_num=[channel_num[0], channel_num[1], channel_num[2]])
        self.attention_1_3 = BlockViTStage3(opt, vis, channel_num=[channel_num[1], channel_num[2], channel_num[3]])
        self.attention_1_4 = BlockViTStage4(opt, vis, channel_num=[channel_num[2], channel_num[3]])

        self.norm1 = nn.LayerNorm(channel_num[0], eps=1e-6)
        self.norm2 = nn.LayerNorm(channel_num[1], eps=1e-6)
        self.norm3 = nn.LayerNorm(channel_num[2], eps=1e-6)
        self.norm4 = nn.LayerNorm(channel_num[3], eps=1e-6)

        self.reconstruct1_1 = Reconstruct1(channel_num[0], channel_num[0], kernel_size=1, scale_factor=2)
        self.reconstruct1_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=2)
        self.reconstruct1_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=2)
        self.reconstruct1_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=2)

    def forward(self, en1, en2, en3, en4):

        emb1_1 = self.embeddings_stage_1_1(en1)
        emb1_2 = self.embeddings_stage_1_2(en2)

        emb2_1 = self.embeddings_stage_2_1(en1)
        emb2_2 = self.embeddings_stage_2_2(en2)
        emb2_3 = self.embeddings_stage_2_3(en3)

        emb3_1 = self.embeddings_stage_3_1(en2)
        emb3_2 = self.embeddings_stage_3_2(en3)
        emb3_3 = self.embeddings_stage_3_3(en4)

        emb4_1 = self.embeddings_stage_4_1(en3)
        emb4_2 = self.embeddings_stage_4_2(en4)

        emb1 = self.attention_1_1(emb1_1, emb1_2)
        emb2 = self.attention_1_2(emb2_1, emb2_2, emb2_3)
        emb3 = self.attention_1_3(emb3_1, emb3_2, emb3_3)
        emb4 = self.attention_1_4(emb4_1, emb4_2)

        emb1 = self.norm1(emb1)
        emb2 = self.norm2(emb2)
        emb3 = self.norm3(emb3)
        emb4 = self.norm4(emb4)

        en1_out = self.reconstruct1_1(emb1)
        en2_out = self.reconstruct1_2(emb2)
        en3_out = self.reconstruct1_3(emb3)
        en4_out = self.reconstruct1_4(emb4)

        return en1_out, en2_out, en3_out, en4_out


class ChannelTransformer(nn.Module):
    def __init__(self, opt, vis=None, img_size=256, in_channels=(64, 128, 256, 512), channel_num=(64, 128, 256, 512),
                 patch_size=(7, 3), strize_size=(4, 2, 1)):
        super(ChannelTransformer, self).__init__()

        self.stage1 = ChannelTransformerStage(opt=opt, vis=vis, img_size=img_size, in_channels=in_channels,
                                              channel_num=channel_num, patch_size=patch_size, strize_size=strize_size)
        self.stage2 = ChannelTransformerStage(opt=opt, vis=vis, img_size=img_size, in_channels=in_channels,
                                              channel_num=channel_num, patch_size=patch_size, strize_size=strize_size)
        self.stage3 = ChannelTransformerStage(opt=opt, vis=vis, img_size=img_size, in_channels=in_channels,
                                              channel_num=channel_num, patch_size=patch_size, strize_size=strize_size)
        self.stage4 = ChannelTransformerStage(opt=opt, vis=vis, img_size=img_size, in_channels=in_channels,
                                              channel_num=channel_num, patch_size=patch_size, strize_size=strize_size)

    def forward(self, en1, en2, en3, en4):
        en1, en2, en3, en4 = self.stage1(en1, en2, en3, en4)
        en1, en2, en3, en4 = self.stage2(en1, en2, en3, en4)
        en1, en2, en3, en4 = self.stage3(en1, en2, en3, en4)
        en1_out, en2_out, en3_out, en4_out = self.stage4(en1, en2, en3, en4)

        return en1_out, en2_out, en3_out, en4_out
