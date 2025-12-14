from datasets import load_dataset
import itertools
import time
from torch.utils.data import Dataset , DataLoader
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.conversation import conv_templates, SeparatorStyle
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
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from PIL import Image
import torch
from tqdm import tqdm
from pprint import pprint
from word2number import w2n
import argparse
import json
import requests
from io import BytesIO
import os
import re
import random
import numpy as np
import torch.nn as nn

class TDIUC_Dataset(Dataset):
    def __init__(self , img_dir , ques_path , ann_path , sample_num):
        questions = sorted(
            json.load(open(ques_path))['questions'],
            key=lambda x : x['question_id']
        )
        annotations = sorted(
            json.load(open(ann_path))['annotations'],
            key=lambda x : x['question_id']
        )

        all_data = []
        for question , annotation in zip(questions , annotations):
            data = {
                'question_id' : question['question_id'],
                'image_id' : question['image_id'],
                'question' : question['question'],
                'question_type' : annotation['question_type'],
                'answer' : annotation['answers'],
                'source' : annotation['ans_source']
            }
            all_data.append(data)
        
        # not eval 'absurd' task
        target_data = [item for item in all_data if item['question_type'] != 'absurd']

        if sample_num > len(target_data):
            print('='*10)
            print(f'the num of TDIUC data is {len(target_data)} , can\'t sample {sample_num}')
            sample_num = len(target_data)
        print(sample_num, f" sample_num")

        self.sample_num = sample_num
        indices = random.sample(range(len(target_data)) , sample_num)
        self.data = [target_data[idx] for idx in indices]
        self.img_dir = img_dir

    def __getitem__(self , idx):
        the_data = self.data[idx]
        question = the_data['question']
        answer = the_data['answer'][0]['answer']
        img_id = str(the_data['image_id'])
        img_filename = 'COCO_val2014_%s.jpg' %(img_id.rjust(12,'0'))
        img_path = os.path.join(self.img_dir , img_filename)
        return img_path , question , answer

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    # print(batch)
    assert len(batch) == 1 , 'batch_size must be 1 !!!'
    img_path , question , answer = batch[0]
    return img_path , question , answer

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def get_sentences(
    tokenizer , model , model_name , image_processor , 
    question , answer , img_path , conv_mode
):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in question:
        if model.config.mm_use_im_start_end:
            question = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
        else:
            question = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
    else:
        if model.config.mm_use_im_start_end:
            question = image_token_se + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
    
    if "llama-2" in model_name.lower():
        conv_m = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_m = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_m = "mpt"
    elif "llama3" in model_name.lower():
        conv_m = 'llama'
    elif "phi" in model_name.lower():
        conv_m = "phi3"
    else:
        conv_m = "llava_v0"

    if conv_mode is not None and conv_m != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_m, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_m

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(img_path).convert("RGB")
    images = []
    images.append(image)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
    ).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    output = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    output = output.strip()
    if output.endswith(stop_str):
        output = str(output[: -len(stop_str)])
    output = output.strip()

    if isNumberWords(answer):
        answer = str(w2n.word_to_num(answer))

    if isNumberWords(output): # Uniform number or word
        output = str(w2n.word_to_num(output))
    else: # Uniform uppercase and lowercase letters
        output = "".join(output[:1].lower() + output[1:]).strip().replace('\n', '')
        answer = "".join(answer[:1].lower() + answer[1:]).strip().replace('\n', '')

    return output , answer

def isNumberWords(x):
    try:
        x = w2n.word_to_num(x)
        return True
    except:
        return False

def set_seed(seed=234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="liuhaotian/llava-v1.5-7b-lora, full-ft:liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, help='the path to base llm')
    parser.add_argument("--model_type", type=str, default="llava")
    parser.add_argument("--img_dir", type=str, default="./multimodal-datasets/TDIUC/Images/val2014/")
    parser.add_argument("--ques_path", type=str, default="./multimodal-datasets/TDIUC/Questions/tenth_filtered_val2014_questions.json")
    parser.add_argument("--ann_path", type=str, default="./multimodal-datasets/TDIUC/Annotations/tenth_filtered_val2014_annotations.json")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--perturbation_params_path" , type=str , default='None' , help='path of perturbated model')
    # add for remove some layers
    parser.add_argument('--remove_layers' , nargs='+')
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    set_seed(seed=234)
    prompt = 'Answer the question in one word.'
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    device = torch.device("cuda")
    tokenizer , model , image_processor , _ = load_pretrained_model(
        args.model_path,
        args.model_base if args.model_base != 'None' else None, 
        model_name,
        args.model_type,
        device=device
    )
    print(model)
    # perturbate model
    if args.perturbation_params_path != "None":
        counter = 0
        model_dict = model.state_dict()
        perturbation_model_dict = torch.load(args.perturbation_params_path)
        for key in perturbation_model_dict:
            new_key = 'model.' + key
            if new_key in model_dict:
                counter += 1
                model_dict[new_key] = perturbation_model_dict[key]
        model.load_state_dict(model_dict)
        print("==========Loading perturbated model==========")
        print(f'{counter} layers are changed')

    # need to remove some layers
    if args.remove_layers:
        new_module_list = nn.ModuleList(
            [module for idx , module in enumerate(model.model.layers) if str(idx) not in args.remove_layers]
        )
        model.model.layers = new_module_list
        print("==========Some layers removed==========")

    tdiuc_dataset = TDIUC_Dataset(
        args.img_dir,
        args.ques_path,
        args.ann_path,
        args.sample_num,
    )

    dataloader = DataLoader(
        dataset=tdiuc_dataset,
        sampler=InferenceSampler(len(tdiuc_dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    the_logger = []
    for batch in tqdm(dataloader):
        img_path , question , answer = batch
        output , answer = get_sentences(
            tokenizer,
            model,
            model_name,
            image_processor,
            question + ' ' + prompt,
            answer,
            img_path,
            args.conv_mode
        )
        the_logger.append({
                'output': output,
                'answer': answer
        })

    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(the_logger))
    
    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating TDIUC ...")


        acc = 0
        for item in merged_outputs:
            if item['output'] == item['answer']:
                acc += 1
        print(f'Accuracy: {round(100*acc/len(merged_outputs) , 2)}%')
    torch.distributed.barrier()


