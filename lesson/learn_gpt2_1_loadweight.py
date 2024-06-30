
from learn_gpt2 import GPT
import torch
from torch.nn import functional as F


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

