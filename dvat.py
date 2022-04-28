# VAT_D
# Copyright (c) 2022-present NAVER Corp.
# Apache License v2.0


import torch.nn as nn
import torch.nn.functional as F

from utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DVAT(nn.Module):

    def __init__(self, args):

        super(DVAT, self).__init__()
                
        self.tsa = args.tsa
        self.confidence = args.confidence
        self.sharpening = args.sharpening
        self.topk = args.topk
        self.swap_ratio = args.swap_ratio
        self.total_steps = args.epochs * args.val_iteration
        self.normalize_grad = args.normalize_grad
    
    def forward(self, model, inputs_s, targets_s, inputs_u, train_step, model_lm):
        
        # tokens to embeds

        src_tokens = inputs_u['input_ids'].clone()
        src_embeds = model.bert.bert.embeddings.word_embeddings(src_tokens).clone()
        src_embeds = src_embeds.detach().requires_grad_(True)
        inputs_u = tokens_to_embeds(inputs_u, inputs_u, src_embeds)

        # model forward & sharpening logit
        
        pred = model(inputs_u)
        pred_log_p = F.log_softmax(pred, dim=-1)
        pred_p = torch.softmax(pred.clone().detach(), dim=1)
        pred_s = pred_p ** (1/self.sharpening)
        pred_s = pred_s / pred_s.sum(dim=1, keepdim=True)

        # calculate adversarial direction (first-order approx. with sharpened label dist.)
        
        adv_loss = F.kl_div(pred_log_p, pred_s, None, None, reduction='batchmean')
        delta_grad = torch.autograd.grad(adv_loss, src_embeds, only_inputs=True)[0]      
        delta_grad = delta_grad.detach() # B S H 
        src_embeds = src_embeds.detach() # B S H

        # discard gradients

        model.zero_grad() 

        # normalize gradients & get VAT-D inputs

        delta_grad = project(delta_grad, norm_type=self.normalize_grad)
        embedding_matrix = model.bert.bert.embeddings.word_embeddings.weight.clone().detach()
        adv_tokens = self.discrete_vat(
                            delta_grad, embedding_matrix, src_tokens, src_embeds, inputs_u, model_lm
        )
                
        adv_inputs_u = {}
        adv_inputs_u = embeds_to_tokens(adv_inputs_u, inputs_u, adv_tokens)
        
        pred_hat = model(adv_inputs_u) 
        logp_hat = F.log_softmax(pred_hat, dim=-1)

        # consistency loss

        loss_const = F.kl_div(logp_hat, pred_s, None, None, reduction='batchmean') 

        # cross-entropy loss

        logits_x = model(inputs_s)
        loss_ce = -1 * torch.sum(F.log_softmax(logits_x, dim=1) * targets_s, dim=1)

        # apply tsa

        if self.tsa is not None:
            tsa_thresh = get_tsa_thresh(self.tsa, train_step, self.total_steps, start=1./logits_x.shape[-1], end=1)
            sup_mask = torch.sum(F.softmax(logits_x, dim=-1) * targets_s,dim=1) < tsa_thresh
            sup_mask = sup_mask.float()
            loss_ce = torch.sum(loss_ce * sup_mask, dim=-1) / torch.max(torch.sum(sup_mask, dim=-1),
                                                                        torch.tensor(1.).to(device))
        else:
            loss_ce = torch.mean(loss_ce)

        return loss_ce, loss_const


    def discrete_vat(self, delta_grad, embedding_matrix, src_tokens, src_embeds, inputs_u, model_lm):
        
        """ 
        Code heavily inspired from Paul Michel 
        https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
        """
        
        # hotflip

        src_embeds = src_embeds.clone().detach() # B S H

        new_embed_dot_grad  = torch.einsum(
                "bij,kj->bik", (delta_grad, embedding_matrix)
        )   
        prev_embed_dot_grad = torch.einsum(
                "bij,bij->bi", (delta_grad, src_embeds)
        )   

        dir_dot_grad = prev_embed_dot_grad.unsqueeze(-1) - new_embed_dot_grad # B S V        
        dir_dot_grad *= -1
        dir_norm = pairwise_distance(src_embeds, embedding_matrix)
        dir_dot_grad /= dir_norm

        # get tokens to perturb

        no_special_tokens = (src_tokens >= 999).float() # no special tokens for perturbation (BERT ver.)
        mask_idx = src_tokens.clone().data.fill_(1.0)
        mask_idx = mask_idx * no_special_tokens
        
        rand = torch.rand(src_tokens.size()).to(device)
        rand = rand > (1-self.swap_ratio)    
        rand = mask_idx * rand.long()
        mask_idx = rand.clone()

        mask_idx = mask_idx.clamp(0, 1)
    
        # LM filtering part

        inputs_u['inputs_embeds'] = src_embeds

        with torch.no_grad():
            pred_lm = model_lm(inputs_u)
        
        _, top_k_lm_idx = torch.topk(pred_lm, dim=2, k=self.topk)
        top_k_lm_idx *= inputs_u['attention_mask'].unsqueeze(-1)
        mask = dir_dot_grad.clone().data.fill_(1.0)
        mask *= -np.inf
        mask.scatter_(2, top_k_lm_idx, 0)
        filtered_dir_dot_grad = dir_dot_grad.clone()
        filtered_dir_dot_grad += mask

        filtered_dir_dot_grad[:,:,:999] = -np.inf # no special tokens as candidates      
        filtered_dir_dot_grad.scatter_(2, src_tokens.unsqueeze(-1), -np.inf) 

        _, adv_flip = filtered_dir_dot_grad.max(2)

        mask_idx = mask_idx.clamp(0, 1)
        
        ori_tokens = src_tokens.clone()
        ori_tokens = ori_tokens * (1-mask_idx)    
        adv_tokens = adv_flip * mask_idx 
        adv_tokens = ori_tokens + adv_tokens
        adv_tokens = adv_tokens.long()

        return adv_tokens


def pairwise_dot_product(src_embeds, vocab_embeds, cosine=False):
    """Compute the cosine similarity between each word in the vocab and each
    word in the source
    If `cosine=True` this returns the pairwise cosine similarity"""
    # Normlize vectors for the cosine similarity
    if cosine:
        src_embeds = F.normalize(src_embeds, dim=-1, p=2)
        vocab_embeds = F.normalize(vocab_embeds, dim=-1, p=2)
    # Take the dot product
    dot_product = torch.einsum("bij,kj->bik", (src_embeds, vocab_embeds))
    return dot_product


def pairwise_distance(src_embeds, vocab_embeds, squared=False):
    """Compute the euclidean distance between each word in the vocab and each
    word in the source"""
    # We will compute the squared norm first to avoid having to compute all
    # the directions (which would have space complexity B x T x |V| x d)
    # First compute the squared norm of each word vector
    vocab_sq = vocab_embeds.norm(p=2, dim=-1) 
    vocab_sq_norm = vocab_sq ** 2
    src_sq = src_embeds.norm(p=2, dim=-1)
    src_sq_norm = src_sq ** 2
    # Take the dot product
    dot_product = pairwise_dot_product(src_embeds, vocab_embeds)
    # Reshape for broadcasting
    # 1 x 1 x |V|
    vocab_sq_norm = vocab_sq_norm.unsqueeze(0).unsqueeze(0)
    # B x T x 1
    src_sq_norm = src_sq_norm.unsqueeze(2)
    # Compute squared difference
    sq_norm = vocab_sq_norm + src_sq_norm - 2 * dot_product
    # Either return the squared norm or return the sqrt
    if squared:
        return sq_norm
    else:
        # Relu + epsilon for numerical stability
        sq_norm = F.relu(sq_norm) + 1e-20
        # Take the square root
        return sq_norm.sqrt()
