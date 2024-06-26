from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ---------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # 为了方便的将emb均匀分配到每个head上，让head们各自学习自己的信息
        # 在一个batch内一次处理所有head的qkv矩阵
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出映射
        self.c_proj = nn.Linear(config.n_embd,  config.n_embd)
        # 正则化
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask参数，只是为了遵循OpenAI的规则，而命名为bias
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # B=BatchSize, T=TokenLength, C=Channel
        qkv = self.c_attn(x)
        q, k , v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        # 包含全部q和k的 attention 矩阵
        att = (q @ k.transpose(-2,-1)) * (1.0/ math.sqrt(k.size(-1))) # 计算qk矩阵的内积
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf')) # 将下三角区域mask起来，赋值为无穷小
        att = F.softmax(att, dim=-1)
        y = att @ v # [B, nh, T, T]x[B, nh, T, hs] -> [B, nh, T, hs] softmax结果与V矩阵做内积
        y = y.transpose(1,2).contiguous().view(B, T, C) #将全部的head一个个concat起来
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module): # transformer 结构
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # 序列最大长度
    vocab_size: int = 50257 # token词表大小：50,000 BPE meges + 256 bytes tokens + 1 <endoftex> token
    n_layer: int = 12 # attention 层数
    n_head: int = 12 # head 数
    n_embd: int = 768 # embd大小

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict(dict( # 为了能直接加载huggingface上下载下来的模型参数文件，需要与其保持结构和变量名的一致性
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embd
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embd
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx 维度 [B, T]
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 前向传播
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # [T]
        pos_emb = self.transformer.wpe(pos) # [T, n_emb]
        tok_emb = self.transformer.wte(idx) # [B, T, n_emb]
        x = tok_emb + pos_emb
        # 依次串联block
        for block in self.transformer.h:
            x = block(x)
        # 加入最后的layernorm和分类
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # 计算交叉熵
        return logits, loss



    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd= 768), # 124M params
            'gpt2-medium':  dict(n_layer=12, n_head=12, n_embd= 768), # 350M params
            'gpt2-large':   dict(n_layer=12, n_head=12, n_embd= 768), # 774M params
            'gpt2-xl':      dict(n_layer=12, n_head=12, n_embd= 768)  # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args) # 将config_args的key和value直接作为参数传入GPTConfig
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 将除了包含‘.attn.bias’外的其他参数作为sd_keys，为啥呢？

        # 初始化 huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 在保证所有参数的‘名称’和‘维度’都对齐的情况下，将参数拷贝过来。
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn_masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 因openai的checkpoint使用的是‘Conv1D’module，但是这里我们使用一个常规的linear，所以这里需要做下转换
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对于 Conv1D 的 weight 特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 从其它参数拷贝
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ------------------------------------------------------------------------
device= "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
print(f"using device: {device}")

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig()) # 如果是随机初始化的参数，则输出结果也会是无意义的
model.eval()
model.to(device)
print("didn't crash yey")


import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
# tokens = enc.encode("你好，我是一个语言模型")
tokens = torch.tensor(tokens, dtype=torch.long )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # 将1D的tensor扩展到2D，为了增加batchsize维度，虽然这里batchsize=1；同时，也复制‘num_return_sequences’份，以产生多个返回结果

x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: # 未达到最大长度之前，一直生成next token
    # model 推理，计算出logit
    with torch.no_grad(): # 确保每次推理，都清理掉上次推理残留的梯度
        logits = model(x)
        # print(logits)
        logits = logits[0][:, -1, :] # 取出返回结果中的【最后一个token】（也就是即将生成的token）的logits
        probs = F.softmax(logits, dim = -1) # 计算所有token emb库中，softmax后的全部结果
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # huggingface 默认topK=50
        ix = torch.multinomial(topk_probs, 1) # 从多项式分布中，提取1个样本
        xcol = torch.gather(topk_indices, -1, ix) # 获取胜选的token
        x = torch.cat((x, xcol), dim=-1) #将生成的新的token加入x中

for i in range(num_return_sequences): # 获取每一个返回的句子里全部的token，并解码成字符串
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
