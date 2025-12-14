from torch.utils.data import Dataset
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.conversation import conv_templates
from bunny.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from bunny.util.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path
)
from collections import OrderedDict
import torch.nn as nn
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pprint import pprint
import argparse
import json
import os
import re
import random
import warnings
import numpy as np

def set_seed(seed=234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
F_IMG_DIR = "./multimodal-datasets/Flickr30k/flickr30k-images"
F_ANN_PATH = "./multimodal-datasets/Flickr30k/results_20130124.token"
class Flickr30k_Image_Dataset(Dataset):
    def __init__(self , img_dir , ann_path , sample_num):
        anns = pd.read_table(ann_path , sep='\t' , header=None , names=['image' , 'caption'])
        img_list = anns['image'][::5].tolist()
        caption_list = anns['caption'][::5].tolist()
        all_img_names = [img_name.split('#')[0] for img_name in img_list]
        assert len(caption_list) == len(all_img_names)

        if sample_num > len(caption_list):
            print('='*10)
            print(f'the num of flickr30k data is {len(caption_list)} , can\'t sample {sample_num}')
            sample_num = len(caption_list)

        random.seed(234)
        random_indices = random.sample(range(len(all_img_names)), sample_num)
        self.img_names = [all_img_names[i] for i in random_indices]
        self.captions = [caption_list[i] for i in random_indices]
        self.img_dir = img_dir
    
    def __getitem__(self , idx):
        img_name = self.img_names[idx]
        caption = self.captions[idx]
        return os.path.join(self.img_dir , img_name) , caption
    
    def __len__(self):
        return len(self.img_names)


def get_hidden_states(
    tokenizer , model , model_name , img_processor , 
    prompt , img_path , device
):
    img = Image.open(img_path).convert("RGB")
    img_tensor = process_images(
        [img],
        img_processor,
        model.config
    )[0].to(device, dtype=torch.float16)

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in prompt:
        if model.config.mm_use_im_start_end:
            prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
        else:
            prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
    else:
        if model.config.mm_use_im_start_end:
            prompt = image_token_se + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" or "llava" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif"llama3" in model_name.lower():
        conv_mode = "llama"
    else:
        conv_mode = "llava_v0"
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)
    prompt_ids = (
        tokenizer_image_token(
            prompt, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0)
    ).to(device)
    print(prompt_ids.shape,prompt_ids[0][35]) # (1,50)
    with torch.no_grad():
        outputs = model(
            prompt_ids, 
            images=img_tensor.unsqueeze(0), 
            output_hidden_states=True
        )
    # print(len(outputs.hidden_states)) # 33
    print(outputs.hidden_states[0].shape) # [1, 625, 4096]
    return outputs.hidden_states


def calc_blockInfluence(hidden_a , hidden_b):
    assert hidden_a.shape == hidden_b.shape 
    # (1, token_num , 4096)
    cosine_like = torch.tensor([0.0], dtype=torch.float32).to(device)
    idx = 0
    for a_t , b_t in zip(hidden_a , hidden_b):
        dot_product = torch.dot(
            a_t.to(dtype=torch.float32), 
            b_t.to(dtype=torch.float32)
        )
        # norm_a_t = torch.norm(a_t)
        # norm_b_t = torch.norm(b_t)
        norm_a_t = torch.norm(a_t , p=2)
        norm_b_t = torch.norm(b_t , p=2)
        tmp = dot_product / (norm_a_t * norm_b_t)
        if torch.isnan(tmp):
            continue
        cosine_like = (idx / (idx+1) * cosine_like) + (tmp / (idx+1))
        idx += 1
    return 1 - cosine_like


# cur_input_embeds_no_im[i].shape torch.Size([35, 4096]) text
# cur_image_features.shape torch.Size([576, 4096]) img
# cur_input_embeds_no_im[i].shape torch.Size([14, 4096]) text
def text_img_calc_blockInfluence(hidden_a , hidden_b):
    assert hidden_a.shape == hidden_b.shape 
    text_hidden_a = torch.cat((hidden_a[:32] , hidden_a[-12:]))
    image_hidden_a = hidden_a[32:-12]
    text_hidden_b = torch.cat((hidden_b[:32] , hidden_b[-12:]))
    image_hidden_b = hidden_b[32:-12]

    # for text
    cosine_like_text = torch.tensor([0.0], dtype=torch.float32).to(device)
    idx = 0
    for a_t , b_t in zip(text_hidden_a , text_hidden_b):
        dot_product = torch.dot(
            a_t.to(dtype=torch.float32), 
            b_t.to(dtype=torch.float32)
        )
        norm_a_t = torch.norm(a_t , p=2)
        norm_b_t = torch.norm(b_t , p=2)
        tmp = dot_product / (norm_a_t * norm_b_t)
        if torch.isnan(tmp):
            continue
        cosine_like_text = (idx / (idx+1) * cosine_like_text) + (tmp / (idx+1))
        idx += 1

    # for image
    cosine_like_image = torch.tensor([0.0], dtype=torch.float32).to(device)
    idx = 0
    for a_t , b_t in zip(image_hidden_a , image_hidden_b):
        dot_product = torch.dot(
            a_t.to(dtype=torch.float32), 
            b_t.to(dtype=torch.float32)
        )
        norm_a_t = torch.norm(a_t , p=2)
        norm_b_t = torch.norm(b_t , p=2)
        tmp = dot_product / (norm_a_t * norm_b_t)
        if torch.isnan(tmp):
            continue
        cosine_like_image = (idx / (idx+1) * cosine_like_image) + (tmp / (idx+1))
        idx += 1

    return 1 - (cosine_like_image + cosine_like_text)/2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b-lora", 
        help="liuhaotian/llava-v1.5-7b-lora, full-ft:liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5" , help='None , lmsys/vicuna-7b-v1.5')
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=1000)
    args = parser.parse_args()
    set_seed(seed=234)
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer , model , img_processor , _ = load_pretrained_model(
        args.model_path,
        args.model_base if args.model_base != 'None' else None, 
        model_name,
        args.model_type,
    )

    device = torch.device("cuda")
    model = model.to(device)
    

    # new_module_list = nn.ModuleList(
    #     [module for idx , module in enumerate(model.model.layers) if idx not in [0,1,2,3,4,5,6,7]]
    # )
    # model.model.layers = new_module_list
    # print('Remove layers !!!!!')

    flickr_dataset = Flickr30k_Image_Dataset(
        F_IMG_DIR,
        F_ANN_PATH,
        args.sample_num
    )

    prompt = "Please describe this image in detail."
    all_bi = OrderedDict()

    for idx , item in tqdm(enumerate(flickr_dataset) , total=len(flickr_dataset)):
        img_path , caption = item
        hidden_states = get_hidden_states(
            tokenizer , model , model_name , img_processor , 
            prompt , img_path , device
        )
        for layer_idx in range(32):
            layer_name = f'layer{layer_idx}'
            bi = calc_blockInfluence(
                hidden_states[layer_idx].squeeze(0), # before layer_i
                hidden_states[layer_idx+1].squeeze(0), # after layer_i
            )
            if layer_name in all_bi:
                all_bi[layer_name] = (idx / (idx+1)) * all_bi[layer_name] + (bi / (idx+1))
            else:
                all_bi[layer_name] = bi

    print('block influence result:')
    pprint(all_bi)
    print('After sort by bi_matrix:')
    sorted_bi = OrderedDict(sorted(all_bi.items() , key=lambda x:x[1]))
    pprint(sorted_bi)

