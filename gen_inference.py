from PIL import Image
import torch
import numpy as np
from options.test_options import TestOptions
from models.BigGAN_networks import Generator
from util.util import prepare_z_y

def load_model():
    opt = TestOptions().parse()  # get test options
    opt.n_classes = 1
    gen = Generator(**vars(opt))

    state_dict = torch.load('./checkpoints/latest_net_G.pth')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    gen.eval()
    return gen

model = load_model()

char_to_int = {
    '妾': 0,
}

def get_word(word):
    encoded = [char_to_int[char] for char in word]
    words = torch.zeros((1, len(encoded), 1), dtype=torch.int32)
    for i, code in enumerate(encoded):
        words[0, i, code] = 1
    return words

def generate_image(word):
    seed = np.random.randint(0, 10e4)
    words = get_word(word)
    z, _ = prepare_z_y(1, 128, 80, device='cpu', seed=seed)
    res = model.forward(z=z, y=words)
    res = res.detach().numpy()[0, 0] * 255
    im = Image.fromarray(res).convert('RGB')
    return im

img = generate_image('妾妾妾妾妾妾')
img.show()