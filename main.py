
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
                    dim_ffn=1024,
                    diffusion_step=1)

weights = torch.load('./gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)['wte.weight']
config.use_embedding = weights
config.vocab_size = weights.size()[0]
config.embedding_dim = weights.size()[1]

source = """[BOS]Sapporo[a] (札幌市, Sapporo-shi, IPA: [sapːoɾo ɕi]) (Ainu: サッ・ポロ・ペッ, romanized: Satporopet, lit. 'Dry, Great River')[2] is a city in Japan. It is the largest city north of Tokyo and the largest city on Hokkaido, the northernmost main island of the country. It ranks as the fifth most populous city in Japan. It is the capital city of Hokkaido Prefecture and Ishikari Subprefecture. Sapporo lies in the southwest of Hokkaido, within the alluvial fan of the Toyohira River, which is a tributary stream of the Ishikari. It is considered the cultural, economic, and political center of Hokkaido.As with most of Hokkaido, the Sapporo area was settled by the indigenous Ainu people, beginning over 15,000 years ago. Starting in the late 19th century, Sapporo saw increasing settlement by Yamato migrants. Sapporo hosted the 1972 Winter Olympics, the first Winter Olympics ever held in Asia, and the second Olympic games held in Japan after the 1964 Summer Olympics. Sapporo is currently bidding for the 2030 Winter Olympics.[3] The Sapporo Dome hosted three games during the 2002 FIFA World Cup and two games during the 2019 Rugby World Cup. Additionally, Sapporo has hosted the Asian Winter Games three times, in 1986, 1990, and 2017 and the 1991 Winter Universiade.The annual Sapporo Snow Festival draws more than 2 million tourists from abroad.[4] Other notable sites include the Sapporo Beer Museum, which is the only beer museum in Japan,[5] and the Sapporo TV Tower located in Odori Park. It is home to Hokkaido University, just north of Sapporo Station. The city is served by Okadama Airport and New Chitose Airport in nearby Chitose.
""".replace("\n", "[SEP]").replace(".", "[BOS]")

target = source

start_sentence = """[BOS]"""

bpe_tokenizer = encoder.get_encoder()

model  = SaltGPT(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

x_sentence = bpe_tokenizer.encode(source)
y_sentence = bpe_tokenizer.encode(target)
s_sentence = bpe_tokenizer.encode(start_sentence)

x_sentence, y_sentence = padding(x_sentence, y_sentence)
y_sentence, s_sentence = padding(y_sentence, s_sentence)

x = torch.tensor(x_sentence).unsqueeze(0)
y = torch.tensor(y_sentence).unsqueeze(0)
start = torch.tensor(s_sentence).unsqueeze(0)

print(x.size())
print(y.size())
print(start.size())

def train(config, model, optimizer, x, y, use_len):
    model.train()
    optimizer.zero_grad()
    y_out = step_model(config, model, x, start)
    loss = 0.0
    
    for n in range(use_len):
        loss += criterion(y_out[:, n, :], y[:, n+1])
    loss.backward()
    print(f"loss: {loss / use_len}")
    optimizer.step()
    generate_sentence(config, model, "[BOS]Sapporo is")
    return loss.item() / use_len

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

generate_sentence(config, model, "[BOS] Sapporo is")

use_len = 3

for i in tqdm(range(1000)):
    loss_result = train(config, model, optimizer, x, y, use_len)

    if loss_result <= 3.0:
        use_len += 3


while True:
    source = input("> Input Something...")
    generate_sentence(config, model, source)
    

#img = make_dot(y, params=dict(model.named_parameters()))
#img.render("model_structure_v1", format="png")




