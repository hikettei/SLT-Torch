from SLT import CachingMHA
import torch

a = CachingMHA(784, 100, 8)
x = torch.randn([10, 500, 784])

from tqdm import tqdm
#for i in tqdm(range(300)):
#    a.forward(x, x, None, approx=False)

for i in tqdm(range(300)):
    a.forward(x, x, None, approx=True)

