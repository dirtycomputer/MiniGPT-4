import argparse
import gc
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
from PIL import Image
import csv

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

def process_batch(image_text_pairs, model):
    results = []
    with open("result.csv", mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id","label","llm_message"])
        for img_id, gr_img, user_message in image_text_pairs:
            if user_message.find("良性") > -1:
                label = "良性"
            elif user_message.find("恶性") > -1:
                label = "恶性"
            user_message.replace("结节为:良性", "结节为:")
            user_message.replace("结节为:恶性", "结节为:")
            user_message += "__? 请只回答该患者结节是恶性还是良性。"
            chat_state = CONV_VISION.copy()
            img_list = []

            # 处理图像
            llm_message = chat.upload_img(gr_img, chat_state, img_list)
            chat.encode_img(img_list) 
            # 发送文本
            chat.ask(user_message, chat_state)

            # 获取回答
            llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
            
            writer.writerow([img_id,label,llm_message])
            print([img_id,label,llm_message])
            torch.cuda.empty_cache()
    return results.append(img_id,label,llm_message)

# 定义一个函数，根据image_id构建文件路径
def build_image_path(image_id):
    # 这里需要你根据实际情况来构建文件路径
    # 例如，你可能有一个基础路径和一种将image_id转换为文件名的方法
    base_path = '/mnt/cache/zhouenshen/HMBM/samples_test/image/'
    filename = image_id + '.jpg'  # 假设文件名直接是image_id加上.jpg后缀
    return base_path + filename
with open("/mnt/cache/zhouenshen/HMBM/samples_test/filter_cap.json", "r") as json_file: 
    data = json.load(json_file)
# 遍历data中的annotations部分，提取图像和文本对
image_text_pairs = [
    (item['image_id'], Image.open(build_image_path(item['image_id'])), item['caption'])
    for item in data['annotations']
]

# 执行批处理
processed_results = process_batch(image_text_pairs, model)

# 打印结果
for result in processed_results:
    print(result)
