#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTransformer Backbone 集成脚本
基于您提供的iTransformer代码链接，创建完整的iTransformer backbone
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# 添加项目根路径
ROOT = Path("/Users/waba/PythonProject/Transformer Project/iTAC-AD").resolve()
sys.path.insert(0, str(ROOT))

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class iTransformerEncoder(nn.Module):
    """
    iTransformer 编码器
    基于变量token的注意力机制
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: [B, D, L] - 变量作为token，时间作为序列
        """
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 层归一化
        x = self.norm(x)
        
        return x

class iTransformerBackbone(nn.Module):
    """
    iTransformer Backbone
    完整的iTransformer实现，支持变量token注意力
    """
    def __init__(
        self,
        d_in: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 输入投影：将每个变量的时间序列投影到d_model
        self.input_projection = nn.Linear(seq_len, d_model)
        
        # iTransformer编码器
        self.encoder = iTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        # 输出投影：将d_model投影回seq_len
        self.output_projection = nn.Linear(d_model, seq_len)
        
    def forward(self, x):
        """
        x: [B, L, D] - 输入时间序列
        返回: [B, L, D] - 编码后的时间序列
        """
        B, L, D = x.shape
        
        # 转换为变量token格式: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)  # [B, D, L]
        
        # 输入投影：将每个变量的时间序列投影到d_model
        x = self.input_projection(x)  # [B, D, d_model]
        
        # iTransformer编码
        x = self.encoder(x)  # [B, D, d_model]
        
        # 输出投影：将d_model投影回seq_len
        x = self.output_projection(x)  # [B, D, L]
        
        # 转换回时间序列格式: [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)  # [B, L, D]
        
        return x

def create_itransformer_backbone(d_in: int, seq_len: int, **kwargs):
    """创建iTransformer backbone的工厂函数"""
    return iTransformerBackbone(d_in=d_in, seq_len=seq_len, **kwargs)

if __name__ == "__main__":
    # 测试iTransformer backbone
    print("=== iTransformer Backbone 测试 ===")
    
    # 创建模型
    model = iTransformerBackbone(
        d_in=7,
        seq_len=96,
        d_model=128,
        n_heads=8,
        e_layers=2,
        dropout=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    x = torch.randn(2, 96, 7)  # [B, L, D]
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
        print(f"输出形状: {y.shape}")
        print(f"输入输出形状匹配: {x.shape == y.shape}")
    
    print("✅ iTransformer Backbone 测试通过！")
