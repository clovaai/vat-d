# VAT_D
# Copyright (c) 2022-present NAVER Corp.
# Apache License v2.0


import argparse
import logging
import multiprocessing

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from transformers import AdamW, get_linear_schedule_with_warmup
from data_utils import get_data
from model import ClassificationBert, LMBert

from utils import *
from dvat import DVAT

logging.getLogger().setLevel(logging.INFO)

def main(args): 

    set_seed(args.seed)

    # read dataset and build dataloaders

    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels  = get_data(
        data_path=args.data_path, n_labeled_per_class=args.n_labeled, 
        unlabeled_per_class=args.un_labeled, model=args.model_ver
    )
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True
    )
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True
    )
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False
    )
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False
    )

    labeled_trainiter = repeat_dataloader(labeled_trainloader)
    unlabeled_trainiter = repeat_dataloader(unlabeled_trainloader)

    # define the models, set the optimizer
    
    model = ClassificationBert(args.model_ver, n_labels).to(device)
    model_lm = LMBert(args.model_ver).to(device)

    model_lm.eval()
    
    t_total = args.epochs * args.val_iteration

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,  
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logging.info(" | Training with Discrete VAT")
    train_criterion = DVAT(args)
    use_unlabeled = True    
    val_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc = 0
    
    # start training

    for epoch in range(args.epochs):
                
        train(
            args, labeled_trainiter, unlabeled_trainiter, model, optimizer, scheduler,
            train_criterion, epoch, n_labels, model_lm, use_unlabeled
        )

        val_loss, val_acc = validate(
            val_loader, model, val_criterion
        )

        logging.info(" | Train Step : {}  Validation Accuracy : {:.2f}  Validation Loss : {:.2f}".format(
            int((epoch+1) * args.val_iteration), val_acc, val_loss)
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_step = int((epoch+1) * args.val_iteration)
            _, test_acc = validate(
                test_loader, model, val_criterion
            )
            best_test_acc = test_acc

    logging.info(' | Best Performance at Train Step : {}'.format(best_step))
    logging.info(' | Best Validation Accuracy : {:.2f}'.format(best_val_acc))
    logging.info(' | Best Test Accuracy : {:.2f}'.format(best_test_acc))


def train(args, labeled_trainiter, unlabeled_trainiter, model, optimizer, scheduler, criterion, epoch, n_labels ,
          model_lm, use_unlabeled):

    train_step = epoch * args.val_iteration
    
    model.train()

    loss_ce_log = 0
    loss_const_log = 0

    for _ in range(args.val_iteration):

        train_step += 1

        inputs_s, targets_s = next(labeled_trainiter)
        inputs_u = next(unlabeled_trainiter)
        targets_s = torch.zeros(inputs_s['input_ids'].size(0), n_labels).scatter_(1, targets_s.view(-1, 1), 1)
        inputs_s = to_device(inputs_s)
        targets_s = targets_s.to(device)
        inputs_u = to_device(inputs_u)

        if use_unlabeled:
            loss_ce, loss_const = criterion(model, inputs_s, targets_s, inputs_u, train_step, model_lm)
            loss_ce_log += loss_ce.item()
            loss_const_log += loss_const.item()
            loss = loss_const + loss_ce
        else:
            pred = model(inputs_s)
            loss = -1 * torch.sum(F.log_softmax(pred, dim=1) * targets_s, dim=1)
            loss = torch.mean(loss)
            loss_ce_log += loss.item()
            loss_const_log = 100
                  
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def validate(validloader, model, criterion):
    
    model.eval()
    
    with torch.no_grad():

        loss_total = 0
        num_sample = 0
        num_correct = 0

        for _, (inputs, targets) in enumerate(validloader):

            inputs = to_device(inputs)
            targets = targets.to(device)    
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            num_correct += torch.sum(predicted==targets).float()
            loss_total += loss.item() * inputs['input_ids'].shape[0]
            num_sample += inputs['input_ids'].shape[0]

        num_sample = torch.tensor(num_sample).to(device).float()
        acc_total = (num_correct / num_sample) * 100
        loss_total = loss_total / num_sample

    return loss_total, acc_total


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DVAT')

    # Training

    parser.add_argument('--epochs',        type=int, default=60,  help='number of total epochs to run')
    parser.add_argument('--seed',          type=int, default=0,   help='seed')
    parser.add_argument('--batch-size',    type=int, default=8,   help='train batchsize')
    parser.add_argument('--batch-size-u',  type=int, default=24,  help='unlabeled train batchsize')
    parser.add_argument('--model-ver',     type=str, default='bert-base-uncased', help='pretrained model version')
    parser.add_argument('--data-path',     type=str)

    # Optimizer

    parser.add_argument('--learning-rate', type=float,  default=3e-5,  help='LR')
    parser.add_argument('--adam-epsilon',  type=float,  default=1e-8,  help='LR')
    parser.add_argument('--warmup-steps',  type=int,    default=1500,  help='WS')
    parser.add_argument('--weight-decay',  type=float,  default=0.0,   help='WD')
    parser.add_argument('--max-grad-norm', type=float,  default=1.0,   help='MG')

    # Constistency Training

    parser.add_argument('--n-labeled',     type=int,   default=10,    help='number of labeled data')
    parser.add_argument('--un-labeled',    type=int,   default=5000,  help='number of unlabeled data')
    parser.add_argument('--val-iteration', type=int,   default=250,   help='every valid step')
    parser.add_argument('--sharpening',    type=float, default=0.5,   help='temperature for sharpen function')
    parser.add_argument('--tsa',           type=str,   default=None,
                        help='scheduler type for training signal annealing')
    parser.add_argument('--confidence',    type=float, default=0,
                        help='confidence threshold for masking unsupervised loss')

    # VAT_D
    
    parser.add_argument('--use-dvat',    action='store_true', dest='use_dvat',
                        default=False, help='whether to use DVAT for training SSL')
    parser.add_argument('--swap-ratio', type=float, default=0.25, help='swap ratio for perturbing sentence')
    parser.add_argument('--topk',       type=int,   default=10,   help='top K candidates for the hotflip operation')
    parser.add_argument('--normalize-grad',  type=str, default="l1",      help='how to noramlize gradients')    
    
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if n_gpu == 0:
        logging.info(" | Training with CPU only")
        logging.info(" | Training with {} CPU".format(multiprocessing.cpu_count()))

    main(args)
