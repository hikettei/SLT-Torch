
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from pathlib import Path
import sys

sys.path.append(
    "./third_party/GPT-2-PyTorch/"
)

from GPT2 import encoder

from SLT.model import (
    step_model,
    SaltConfig,
    SaltGPT
    )

from tqdm import tqdm

#from torchviz import make_dot

def padding(x, y):
    cond = len(x) >= len(y)
    longer_one = y if cond else x
    diff = abs(len(x) - len(y))
    for i in range(diff):
        longer_one.append(0)
    return (x, longer_one) if cond else (longer_one, y)

config = SaltConfig(opt_forward=False,
                    opt_backward=False,
                    nlayers=3,
                    dim_ffn=512,
                    diffusion_step=3)

weights = torch.load('./gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)['wte.weight']
config.use_embedding = weights
config.vocab_size = weights.size()[0]
config.embedding_dim = weights.size()[1]

source = """[BOS]
Inspired by the relationship between the ODE and neural networks [25, 8], we first show that the
Transformer layers can be naturally interpreted as a numerical ODE solver for a first-order convectiondiffusion equation in MPDS. To be more specific, the self-attention sub-layer, which transforms
the semantics at one position by attending over all other positions, corresponds to the diffusion
term; The position-wise FFN sub-layer, which is applied to each position separately and identically,
corresponds to the convection term. The number of stacked layers in the Transformer corresponds to
the time dimension in ODE. In this way, the stack of self-attention sub-layers and position-wise FFN
sub-layers with residual connections can be viewed as solving the ODE problem numerically using
the Lie-Trotter splitting scheme [17] and the Euler’s method [3]. By this interpretation, we have
a novel understanding of learning contextual representations of a sentence using the Transformer:
the feature (a.k.a, embedding) of words in a sequence can be considered as the initial positions of a
collection of particles, and the latent representations abstracted in stacked Transformer layers can be
viewed as the location of particles moving in a high-dimensional space at different time points
""".replace("\n", "")

target = """[BOS]
Such an interpretation not only provides a new perspective on the Transformer but also inspires us
to design new structures by leveraging the rich literature of numerical analysis. The Lie-Trotter
splitting scheme is simple but not accurate and often leads to high approximation error [17]. The
Strang-Marchuk splitting scheme [39] is developed to reduce the approximation error by a simple
modification to the Lie-Trotter splitting scheme and is theoretically more accurate. Mapped to neural
network design, the Strang-Marchuk splitting scheme suggests that there should be three sub-layers:
two position-wise feed-forward sub-layers with half-step residual connections and one self-attention
sub-layer placed in between with a full-step residual connection. By doing so, the stacked layers will
be more accurate from the ODE’s perspective and will lead to better performance in deep learning.
As the FFN-attention-FFN layer is Macaron-like, we call it Macaron layer and call the network
composed of Macaron layers the Macaron Net.
We conduct extensive experiments on both supervised and unsupervised learning tasks. For each task,
we replace Transformer layers by Macaron layers and keep the number of parameters to be the same.
Experiments show that the Macaron Net can achieve higher accuracy than the Transformer on all
tasks which, in a way, is consistent with the ODE theory.
""".replace("\n", "")


bpe_tokenizer = encoder.get_encoder()

model  = SaltGPT(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

x_sentence = bpe_tokenizer.encode(source)
y_sentence = bpe_tokenizer.encode(target)

x_sentence, y_sentence = padding(x_sentence, y_sentence)

x = torch.tensor(x_sentence).unsqueeze(0)
y = torch.tensor(y_sentence).unsqueeze(0)

print(x.size())
print(y.size())

def train(config, model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    y_out = step_model(config, model, x, y)
    loss = 0.0
    for n in range(y.size(1)):
        loss += criterion(y_out[:, n, :], y[:, n])
    loss.backward()
    print(f"loss: {loss / y.size(1)}")
    optimizer.step()
    generate_sentence(config, model, "[BOS] Inspired by the relationship.")

def generate_sentence(config, model, source, input_more="", sentence_len=50):
    x_first = source
    y_first = "[BOS]" + input_more
    
    x_first = bpe_tokenizer.encode(x_first)
    y_first = bpe_tokenizer.encode(y_first)

    x, y = padding(x_first, y_first)

    for _ in range(sentence_len):
        x.append(0)
        y.append(0)
    
    x = torch.tensor(x).unsqueeze(0)
    y = torch.tensor(y).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        y_out = step_model(config, model, x, y)
        y_decode = torch.argmax(y_out, dim=-1)
        print(bpe_tokenizer.decode(y_decode[0].tolist()))

generate_sentence(config, model, "[BOS] Inspired by the relationship")
    

for i in tqdm(range(10)):
    train(config, model, optimizer, x, y)

while True:
    source = input("> Input Something...")
    generate_sentence(config, model, source)
    

#img = make_dot(y, params=dict(model.named_parameters()))
#img.render("model_structure_v1", format="png")




