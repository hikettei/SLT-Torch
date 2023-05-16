from SLT import CachingMHA
import torch

a = CachingMHA(784, 100, 8)
x = torch.randn([1, 500, 784])
y = torch.randn([1, 400, 784])

from tqdm import tqdm

#for i in tqdm(range(300)):
#    a.forward(x, x, None, approx=False)

#for i in tqdm(range(300)):
#    a.forward(x, x, None, approx=True)

print(a.forward(x, y, None, approx=False))
print(a.forward(x, y, None, approx=True))
