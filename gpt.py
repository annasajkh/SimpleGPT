import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.optim import AdamW
from IPython.display import clear_output

#GLU Variant https://arxiv.org/abs/2002.05202
#SwiGLU https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# NormFormer https://arxiv.org/abs/2110.09456
# HeadScaling https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/modules/transformer_layer.py
class TransformerBlock(nn.Module): 
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_scale=4,
        attn_mask=None,
        attn_drop=0,
        resid_pdrop=0
    ):
        super().__init__()
        self.attn_mask = attn_mask
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop)
        self.scale_attn = nn.Parameter(torch.ones((n_heads,)), requires_grad=True)
        
        self.pre_attn_layer_norm = nn.LayerNorm(d_model)
        self.pre_mlp_layer_norm = nn.LayerNorm(d_model)
        self.post_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_scale * 2, bias=False),
            SwiGLU(),
            nn.LayerNorm(d_model * mlp_scale),
            nn.Linear(d_model * mlp_scale, d_model, bias=False),
            nn.Dropout(resid_pdrop)
        )
    
    def attention(self, x): 
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        
        x = self.attn(x, x, x, attn_mask=self.attn_mask, need_weights=False)[0]
                
        tgt_len, bsz = x.size(0), x.size(1)
        x = x.view(tgt_len, bsz, self.n_heads, self.attn.head_dim)
        x = torch.einsum("tbhd,h->tbdh", x, self.scale_attn)
        x = x.reshape(tgt_len, bsz, self.d_model)
        
        return x
    
    def forward(self, x):
        x = x + self.post_attn_layer_norm(self.attention(self.pre_attn_layer_norm(x)))
        x = x + self.mlp(self.pre_mlp_layer_norm(x))
        
        return x


class Transformer(nn.Module): 
    def __init__(
        self,
        n_context,
        n_embed,
        n_heads,
        n_layers, 
        mlp_scale=4,
        attn_mask=None,
        attn_drop=0,
        resid_pdrop=0,
        embed_pdrop=0
    ):
        super().__init__()
        
        self.drop = nn.Dropout(embed_pdrop) 
        self.pos_embed = nn.Parameter(torch.randn(n_context, n_embed)) 
        self.layers = nn.Sequential(*[TransformerBlock(n_embed, n_heads, attn_mask=attn_mask, mlp_scale=mlp_scale, attn_drop=attn_drop, resid_pdrop=resid_pdrop) for _ in range(n_layers)]) 
        self.ln_pre = nn.LayerNorm(n_embed)

    #the input is an embbeding with this shape (batch, n_context, n_embed) 
    def forward(self, x):
        x = self.drop(x + self.pos_embed)
        x = self.ln_pre(x) 

        x = x.permute(1, 0, 2)
        x = self.layers(x) 
        x = x.permute(1, 0, 2)

        return x


class GPT(nn.Module):
    def __init__(
        self,
        n_heads,
        n_layers,
        n_embed,
        block_size,
        n_vocab,
        ignore_token,
        mlp_scale=4,
        attn_drop=0,
        resid_pdrop=0,
        embed_pdrop=0
    ):
        super().__init__()
        
        self.embed = nn.Embedding(n_vocab, n_embed)
        
        self.transformer = Transformer(n_context=block_size,
                                       n_embed=n_embed,
                                       n_heads=n_heads,
                                       n_layers=n_layers,
                                       mlp_scale=mlp_scale,
                                       attn_mask=torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1),
                                       attn_drop=attn_drop,
                                       resid_pdrop=resid_pdrop,
                                       embed_pdrop=embed_pdrop)
        
        self.ln_post = nn.LayerNorm(n_embed)
        self.out_head = nn.Linear(n_embed, n_vocab, bias=False)
        self.ignore_token = ignore_token
        self.block_size = block_size
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.transformer.pos_embed, mean=0.0, std=0.02)
    
    def forward(self, x, y=None):
        x = self.embed(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        
        logits = self.out_head(x)
        
        loss = None
        
        if y is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=self.ignore_token)
        
        return logits, loss
    
    @torch.no_grad()
    def sample(self, x, temperature=1.0, top_k=40, max_length=100, batch_size=1):
        x = torch.cat([torch.tensor([self.ignore_token, ]).to(device), x])
        
        batch = torch.stack([torch.cat([x, torch.full((self.block_size - len(x),), self.ignore_token).to(device)]) for _ in range(0, batch_size)]).to(x.device)
        
        
        length = self.block_size if len(x) - 1 + max_length > self.block_size else len(x) - 1 + max_length
        
        for i in tqdm(range(len(x) - 1, length)): 
            logits = self(batch)[0][:, i]
            
            logits = F.softmax(logits / temperature, dim=1)
            logits = torch.topk(logits, top_k)

            out = logits[1][:, torch.multinomial(logits[0], num_samples=1)]

            if torch.all(out) and out[0] == self.ignore_token:
                break
            
            batch[:, i + 1] = out

        return batch



device = "cuda" if torch.cuda.is_available() else "cpu"


model = GPT(n_heads=6, n_layers=6, n_embed=768, block_size=1024, n_vocab=50257, ignore_token=50256)
# model.load_state_dict(torch.load("transformer.pkl"))
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
