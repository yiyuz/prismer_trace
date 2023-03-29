# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import argparse
import numpy as np
import random
import time
import functools
import json
import torch
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

import sys
sys.path.append('.')

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from model.prismer_caption import PrismerCaption
from model.modules.utils import interpolate_pos_embed
from dataset import create_dataset, create_loader
from utils import *
from tqdm import tqdm

from pdb import set_trace


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='')
parser.add_argument('--port', default='')

parser.add_argument('--config', default='configs/caption.yaml')
parser.add_argument('--from_checkpoint', action='store_true')
parser.add_argument('--target_dataset', default='nocaps', type=str)
parser.add_argument('--shard_grad_op', action='store_true')
parser.add_argument('--full_shard', action='store_true')
parser.add_argument('--exp_name', default='prismer_large', type=str)
parser.add_argument('--mixed_precision', default='fp16', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
os.environ["PYTHONPATH"] = "."

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.target_dataset]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

test_dataset = create_dataset('eval_caption', config)
test_loader = create_loader(test_dataset, batch_size=config['batch_size_test'], num_workers=4, train=False)
# test_loader = create_loader(test_dataset, batch_size=config['batch_size_test'], num_workers=8, train=False)

model = PrismerCaption(config)
tokenizer = model.tokenizer

if args.shard_grad_op:  # Model Sharding: ZeRO 2
    from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy, StateDictType
    fsdp_plugin = FullyShardedDataParallelPlugin(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                                                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                                                 mixed_precision_policy=MixedPrecision(param_dtype=torch.float16,
                                                                                       reduce_dtype=torch.float16,
                                                                                       buffer_dtype=torch.float16),
                                                 state_dict_type=StateDictType.FULL_STATE_DICT,
                                                 ignored_modules=model.ignored_modules)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare(model)

elif args.full_shard:  # Model Sharding: ZeRO 3
    from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from model.modules.vit import ResidualAttentionBlock
    from model.modules.resampler import PerceiverAttentionBlock
    from model.modules.roberta import RobertaLayer
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            ResidualAttentionBlock,
            PerceiverAttentionBlock,
            RobertaLayer
        },
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(sharding_strategy=ShardingStrategy.FULL_SHARD,
                                                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                                                 mixed_precision_policy=MixedPrecision(param_dtype=torch.float16,
                                                                                       reduce_dtype=torch.float16,
                                                                                       buffer_dtype=torch.float16),
                                                 state_dict_type=StateDictType.FULL_STATE_DICT,
                                                 auto_wrap_policy=auto_wrap_policy,
                                                 ignored_modules=model.ignored_modules)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare(model)
else:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

# Reload saved states
if not args.from_checkpoint:
    state_dict = torch.load(f'logging/pretrain_{args.exp_name}/pytorch_model.bin', map_location='cpu')
    state_dict['expert_encoder.positional_embedding'] = interpolate_pos_embed(state_dict['expert_encoder.positional_embedding'], len(model.expert_encoder.positional_embedding))
    model.load_state_dict(state_dict)
    start_epoch = 0
else:
    state_dict = torch.load(f'logging/caption_{args.exp_name}/pytorch_model.bin', map_location='cuda')
    # state_dict = torch.load(f'logging/caption_{args.exp_name}/pytorch_model.bin', map_location='cpu')
    # if os.path.exists(f'logging/caption_{args.exp_name}/epoch.pt'):
    #     start_epoch = torch.load(f'logging/caption_{args.exp_name}/epoch.pt')[0] + 1
    # else:
    #     start_epoch = 0
    model.load_state_dict(state_dict)
    # accelerator.print(f'Start re-training from checkpoint with Epoch {start_epoch}')

optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config['init_lr'], weight_decay=config['weight_decay'])

best = 0
start_time = time.time()

model.eval()
if accelerator.is_main_process:
    result = []

with torch.no_grad():
    for step, (experts, data_ids) in enumerate(tqdm(test_loader)):
        captions = model(experts, train=False, prefix=config['prefix'])

        if accelerator.use_distributed:
            captions = tokenizer(captions, max_length=30, padding='max_length', return_tensors='pt').input_ids
            captions = captions.to(experts['rgb'].device)
            data_ids, captions = accelerator.gather_for_metrics((data_ids, captions))

        if accelerator.is_main_process:
            for data_id, caption in zip(data_ids, captions):
                caption = tokenizer.decode(caption, skip_special_tokens=True)
                if args.target_dataset == 'coco':
                    image_id = int(test_loader.dataset.data_list[data_id]['image'].split('/')[-1].strip('.jpg').split('_')[-1])
                    result.append({"image_id": image_id, "caption": caption.capitalize() + '.'})
                elif args.target_dataset == 'nocaps':
                    result.append({"image_id": test_loader.dataset.data_list[data_id]['img_id'],
                                   "caption": caption.capitalize() + '.'})

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    json.dump(result, open(f'/results/caption_results_{args.exp_name}_{args.target_dataset}.json', 'w'))
    if args.target_dataset == 'coco':
        coco_caption_eval(f'{config["data_path"]}/coco_karpathy_test_gt.json', result)


