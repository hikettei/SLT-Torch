from SLT import CachingMHA
import torch

a = CachingMHA(784, 100, 8, C=4)
x = torch.randn([1, 1024, 784])
y = torch.randn([1, 1024, 784])

from tqdm import tqdm

for i in tqdm(range(100)):
    a.forward(x, x, None, approx=False)

print(a.forward(x, y, None, approx=True))
    
for i in tqdm(range(100)):
    a.forward(x, x, None, approx=True)

x = torch.randn([1, 1024, 784])
y = torch.randn([1, 1024, 784])

print("approx=False")
print(a.forward(x, y, None, approx=False))
print("approx=True")
print(a.forward(x, y, None, approx=True))
