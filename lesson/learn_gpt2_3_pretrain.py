
from learn_gpt2 import (
    GPTConfig,
    GPT
    )
import torch
from torch.nn import functional as F

# ------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # 从磁盘中加载数据，并放入内存中
        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # 输出
        y = (buf[1:]).view(B, T) # 目标
        # 增加position
        self.current_position += B * T
        # 如果下一个batch超过了闲置的tokens长度，则重置position
        if (self.current_position + (B * T +1) > len(self.tokens)):
            self.current_position = 0
        return x, y



# ------------------------------------------------------------------------
device= "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
print(f"using device: {device}")

model = GPT(GPTConfig())
model.to(device)

train_loader = DataLoaderLite(B=4, T=32)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward() # 前向传播，优化参数
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")


import sys; sys.exit(0)
