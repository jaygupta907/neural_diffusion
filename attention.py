
from einops import rearrange
from torch import nn,einsum
import torch
from einops_exts import check_shape, rearrange_many
from utils import *

class SpatialLinearAttention(nn.Module): # Linear Attention over channels
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # Hidden dimension is the number of heads multiplied by the dimension of each head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False) # Linear layer to get the queries, keys and values
        self.to_out = nn.Conv2d(hidden_dim, dim, 1) # Linear layer to get the output

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w') # Frames are treated as batch to apply spatial attention over channels

        qkv = self.to_qkv(x).chunk(3, dim = 1)  # Divide the channels into three parts : q,k,v : 
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads) # Divide each of q,k and v further into number of heads and flatten each each channel

        q = q.softmax(dim = -2) # Take softmax along the embedding dimension
        k = k.softmax(dim = -1) # Take softmax along the sequence length dimension

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v) # Multiplies K transpose with V

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q) # Multiplies the context with Q
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w) # Rearrange the output to the original shape and combine channels for all heads
        out = self.to_out(out) # Apply a linear layer to get the output
        return rearrange(out, '(b f) c h w -> b c f h w', b = b) # Seperate out frames and batches


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)