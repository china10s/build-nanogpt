
from learn_gpt2 import (
    GPTConfig,
    GPT
    )
import torch
from torch.nn import functional as F

# ------------------------------------------------------------------------
device= "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
print(f"using device: {device}")

import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt','r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4,32 # bach size = 4, 长度 = 32
buf = torch.tensor(tokens[:B*T + 1], device=device)  # 构建需要的B*T大小的tensor
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# 计算logits
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x,y)

# print(loss)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward() # 前向传播，优化参数
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys; sys.exit(0)
