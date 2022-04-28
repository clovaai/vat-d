# VAT_D
# Copyright (c) 2022-present NAVER Corp.
# Apache License v2.0


import random
import numpy as np
import torch

DEFAULT_IGNORE_TOKENS = ["@@NULL@@", ".", ",", ";", "!", "?", "[MASK]", "[SEP]", "[CLS]"]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(inputs):
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)
    return inputs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'lin_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    threshold = threshold * (end - start) + start
    return threshold.to(device)


def get_confidence_mask(confidence_threshold, prob_dist):
    if confidence_threshold != -1: 
            unsup_loss_mask = torch.max(prob_dist, dim=-1)[0] > confidence_threshold
            unsup_loss_mask = unsup_loss_mask.type(torch.float32)
    else:
        unsup_loss_mask = torch.ones(len(prob_dist), dtype=torch.float32)
        unsup_loss_mask = unsup_loss_mask.to(device)

    return unsup_loss_mask


def project(grad, norm_type='inf', eps=1e-6):
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'inf':
        direction = grad.sign()
    return direction


def tokens_to_embeds(empty_dict, orig_dict, new_embeds):
    empty_dict['input_ids'] = None
    empty_dict['inputs_embeds'] = new_embeds
    empty_dict['attention_mask'] = orig_dict['attention_mask'].clone()
    empty_dict['token_type_ids'] = orig_dict['token_type_ids'].clone()
    
    return empty_dict


def embeds_to_tokens(empty_dict, orig_dict, new_tokens):
    empty_dict['input_ids'] = new_tokens
    empty_dict['inputs_embeds'] = None
    empty_dict['attention_mask'] = orig_dict['attention_mask'].clone()
    empty_dict['token_type_ids'] = orig_dict['token_type_ids'].clone()
    
    return empty_dict


def repeat_dataloader(iterable):
    while True:
        for x in iterable:
            yield x