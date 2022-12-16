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


#NormFormer https://arxiv.org/abs/2110.09456
class ConditionedTransformerBlock(nn.Module): 
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
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, bias=False)
        self.conditioned_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, bias=False)
        
        self.pre_attn_layer_norm = nn.LayerNorm(d_model)
        self.post_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.pre_conditioned_attn_layer_norm = nn.LayerNorm(d_model)
        self.post_conditioned_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.x_conditioned_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.pre_mlp_layer_norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_scale * 2, bias=False),
            SwiGLU(),
            nn.LayerNorm(d_model * mlp_scale),
            nn.Linear(d_model * mlp_scale, d_model, bias=False),
            nn.Dropout(resid_pdrop)
        )
        
        self.cached = None
    
    def attention(self, x, cache=None): 
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        
        if cache is not None:
            out = self.attn(x[-1:], x, x, attn_mask=None, need_weights=False)[0]
            return torch.cat([cache, out], dim=0)
        else:
            return self.attn(x, x, x, attn_mask=self.attn_mask[:x.shape[0], :x.shape[0]], need_weights=False)[0]
    
    def forward(self, x, x_conditioned):
        if not self.training:
            attn = self.attention(self.pre_attn_layer_norm(x), cache=self.cached)
            self.cached = attn
        else:
            attn = self.attention(self.pre_attn_layer_norm(x))
         
        x = x + self.post_attn_layer_norm(attn)
        
        x_conditioned = self.x_conditioned_attn_layer_norm(x_conditioned)
        
        conditioned_attn = self.conditioned_attn(self.pre_conditioned_attn_layer_norm(x), x_conditioned, x_conditioned, need_weights=False)[0]
        x = x + self.post_conditioned_attn_layer_norm(conditioned_attn)
        
        x = x + self.mlp(self.pre_mlp_layer_norm(x))
        
        return x


class ConditionedTransformer(nn.Module): 
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
        self.pos_embed = nn.Parameter(torch.randn(1, n_context, n_embed)) 
        self.layers = nn.Sequential(*[ConditionedTransformerBlock(n_embed,
                                                       n_heads,
                                                       attn_mask=attn_mask,
                                                       mlp_scale=mlp_scale,
                                                       attn_drop=attn_drop,
                                                       resid_pdrop=resid_pdrop) for _ in range(n_layers)]) 
        self.ln_pre = nn.LayerNorm(n_embed)
        
    #the input is an embbeding with this shape (batch, n_context, n_embed), x_conditioned is the same
    def forward(self, x, x_conditioned):
        x = self.drop(x + self.pos_embed[:, :x.shape[1], :])
        x = self.ln_pre(x) 

        x = x.permute(1, 0, 2)
        x_conditioned = x_conditioned.permute(1, 0, 2)
        
        for i in range(0, len(self.layers)):
            x = self.layers[i](x, x_conditioned)
        
        x = x.permute(1, 0, 2)
        
        return x


class ConditionedGPT(nn.Module):
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
        
        self.transformer = ConditionedTransformer(n_context=block_size,
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
        elif isinstance(module, ConditionedGPT):
            torch.nn.init.normal_(module.transformer.pos_embed, mean=0.0, std=0.02)
    
    def reset_cache(self):
        for layer in self.transformer.layers:
            layer.cached = None

    def forward(self, x, x_conditioned, y=None):
        x_conditioned = x_conditioned.unsqueeze(1).repeat_interleave(self.block_size, dim=1).to(x.device).float()
        
        x = self.embed(x)
        x = self.transformer(x, x_conditioned)
        x = self.ln_post(x)
        
        logits = self.out_head(x)
        
        loss = None
        
        if y is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=self.ignore_token)


        return logits, loss

    @torch.no_grad()
    def sample(self, x, x_conditioned, temperature=1.0, top_k=40, max_length=100, batch_size=1, repetition_penalty=1.0):
        self.eval()
        self.reset_cache()
        
        batch = torch.stack([torch.cat([torch.tensor([self.ignore_token, ]).to(x.device), x]) for _ in range(0, batch_size)]).to(x.device)
        length = self.block_size if len(x) + max_length > self.block_size else len(x) + max_length
        
        x_conditioned = x_conditioned.repeat_interleave(batch_size, dim=0).to(x.device)
        
        for _ in tqdm(range(len(x), length)):
            logits = self(batch, x_conditioned)[0][:, -1]
            
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_tokens in set(batch[i].tolist()):
                        logits[i, previous_tokens] /= repetition_penalty
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if logits[i, previous_tokens] < 0:
                            logits[i, previous_tokens] *= repetition_penalty
                        else:
                            logits[i, previous_tokens] /= repetition_penalty
            
            logits = F.softmax(logits / temperature, dim=1)
            logits = torch.topk(logits, top_k)

            out = logits[1].gather(1, torch.multinomial(logits[0], num_samples=1))

            if torch.all(out) and out[0] == self.ignore_token:
                break

            batch = torch.cat([batch, out], dim=1)
        
        return batch



device = "cuda" if torch.cuda.is_available() else "cpu"


model = ConditionedGPT(n_heads=6, n_layers=6, n_embed=768, block_size=1024, n_vocab=50257, ignore_token=50256)
# model.load_state_dict(torch.load("transformer.pkl"))
model.to(device)
model.train()
print(model.training)
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.995))
