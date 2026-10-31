"""Compact joint-lead time-patch Transformer frozen for M3."""
from __future__ import annotations
import torch
from torch import nn

class TimePatchTransformer(nn.Module):
    def __init__(self, samples=1000, patch=10, d_model=192, heads=6, blocks=4, mlp=768, dropout=.1):
        super().__init__()
        if samples % patch: raise ValueError("samples must divide exactly into patches")
        self.tokens=samples//patch
        self.proj=nn.Linear(12*patch,d_model)
        self.pos=nn.Parameter(torch.zeros(1,self.tokens,d_model))
        layer=nn.TransformerEncoderLayer(d_model,heads,mlp,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,blocks,norm=nn.LayerNorm(d_model))
        self.head=nn.Linear(d_model,3)
        nn.init.trunc_normal_(self.pos,std=.02)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(12,1000):raise ValueError('expected [batch,12,1000]')
        x=x.reshape(x.shape[0],12,self.tokens,10).permute(0,2,1,3).reshape(x.shape[0],self.tokens,120)
        return self.head(self.encoder(self.proj(x)+self.pos).mean(1))
