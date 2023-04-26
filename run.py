# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import pickle
import csv
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
import time
from model import Seq2Seq
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)
from preprocess import summary_replace, code_replace

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def initParser():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_snippet_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_snippet_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_snippet_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--source_lang", default=None, type=str,
                        help="The language of input")
    parser.add_argument("--target_lang", default=None, type=str,
                        help="The language of input")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_comment_length", default=32, type=int,
                        help="The maximum total comment sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--weightAB', type=float, default=1)
    parser.add_argument('--weightBB', type=float, default=0)
    parser.add_argument('--weightcon', type=float, default=0)
    parser.add_argument('--autonum', type=int, default=0)
    return parser

def read_examples(filename):
    """Read examples from filename."""
    print(len(filename))
    examples, textdict = [], {}
    for file in filename:
        source, target, text, map = file.split(',')
        with open(source, encoding="utf-8") as f1, open(target, encoding="utf-8") as f2, \
                open(text, encoding="utf-8") as f3, open(map, encoding="utf-8") as f4:
            for line in f3.readlines():
                line = json.loads(line)
                if (text.find("program") == -1):
                    summarystr = summary_replace(line["comment_tokens"])
                    #summarystr = line["comment_tokens"]
                    textdict[line["idx"]] = summarystr
                else:
                    summarystr = summary_replace(line["desc_tokens"])
                    #summarystr = line["desc_tokens"]
                    textdict[line["idx"].split('-')[0]] = summarystr
            for line1, line2, line4 in zip(f1, f2, f4):
                line1 = code_replace(line1.strip())
                line2 = code_replace(line2.strip())
                #line1 = line1.strip()
                #line2 = line2.strip()
                line4 = line4.strip()
                comments = textdict[line4]
                examples.append((line1, line2, comments))  # (source_code, target_code, comment)
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, example_id, source_ids, source_mask, target_ids, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask

def convert_examples_to_features(examples, tokenizer, args, stage = None):
    features = []
    for example_index, (source_code, target_code, comment) in enumerate(examples, 0):
        source_tokens = tokenizer.tokenize(source_code)[: args.max_source_length - args.max_comment_length - 2]
        source_codes = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        padding_length = args.max_target_length - len(source_codes)
        source_code_ids = tokenizer.convert_tokens_to_ids(source_codes) + [tokenizer.pad_token_id] * padding_length
        source_code_mask = [1] * len(source_codes) + [0] * padding_length

        target_tokens = tokenizer.tokenize(target_code)[: args.max_target_length - args.max_comment_length - 2]
        target_codes = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        padding_length = args.max_target_length - len(target_codes)
        target_code_ids = tokenizer.convert_tokens_to_ids(target_codes) + [tokenizer.pad_token_id] * padding_length
        target_code_mask = [1] * len(target_codes) + [0] * padding_length

        comment_tokens = tokenizer.tokenize(comment)[: args.max_comment_length - 1]
        comments = comment_tokens + [tokenizer.sep_token]
        padding_length = args.max_comment_length - len(comments)
        comment_ids = tokenizer.convert_tokens_to_ids(comments) + [tokenizer.pad_token_id] * padding_length
        comment_mask = [1] * len(comments) + [0] * padding_length

        features.append(InputFeatures(example_id = example_index,
                                      source_ids = source_code_ids,
                                      source_mask = source_code_mask,
                                      target_ids = target_code_ids,
                                      target_mask = target_code_mask))
    return features

class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask))
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = initParser()
    args = parser.parse_args()
    logger.info(args) # print arguments
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('cuda is not available')
        assert 0
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Set seed
    set_seed(args.seed)

    config = AutoConfig.from_pretrained('/data3/duyl/codetranslation/pretrainmodels/codebert')
    tokenizer = AutoTokenizer.from_pretrained('/data3/duyl/codetranslation/pretrainmodels/codebert')
    encoder = AutoModel.from_pretrained('/data3/duyl/codetranslation/pretrainmodels/codebert')

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id, args=args)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.n_gpu > 1: # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename.split('@')[:-1])
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_data = TextDataset(train_features, args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size, num_workers = 4)
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        dev_dataset, best_bleu, best_loss = {}, 0, 1e6
        for epoch in range(args.num_train_epochs):
            time0 = time.time()
            model.train()
            state, tr_loss = 'train', 0
            if(epoch < args.autonum and args.weightBB!=0):
                state = 'train'
                weightAB = 0.0
                weightBB = args.weightBB
                weightcon = 0.0
            else:
                state = 'finetune'
                weightAB = args.weightAB
                weightBB = args.weightBB
                weightcon = args.weightcon

            for i, batch in enumerate(train_dataloader, 0):
                source_ids, source_mask, target_ids, target_mask = tuple(t.to(device) for t in batch)
                loss = model(state, source_ids, source_mask, target_ids, target_mask, args, weightAB, weightBB, weightcon)
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            time1 = time.time()
            print("%s to %s, epoch %d, batch loss: %.4f, time cost %.2fs" % (args.source_lang, args.target_lang, epoch, tr_loss / i, time1-time0))

            if args.do_eval and state == 'finetune' and (epoch + 1) % 20 == 0:
                #Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.test_filename.split('@'))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)
                model.eval() 
                p, accs =[], []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch
                    with torch.no_grad():
                        preds = model('test', source_ids, source_mask)
                        for pred in preds:
                            t=pred[0].cpu().numpy().tolist()
                            if 0 in t:
                                t=t[:t.index(0)]
                            p.append(tokenizer.decode(t,clean_up_tokenization_spaces=False))
                            
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref, (_, truth,_) in zip(p, eval_examples):
                        f.write(ref + '\n')
                        f1.write(truth + '\n')     
                        accs.append(ref == truth)
                dev_bleu = round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),2)
                xmatch = round(np.mean(accs) * 100, 4)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs)*100,4))))
                if dev_bleu > best_bleu :
                    logger.info("Source %s, Target %s, Best BLEU:%s" % (args.source_lang, args.target_lang, dev_bleu))
                    best_bleu = dev_bleu
                    fcsv = open('result.csv', 'a+', encoding='utf-8')
                    csv_writer = csv.writer(fcsv)
                    csv_writer.writerow(
                        [args.source_lang, args.target_lang, epoch, args.max_source_length,
                         args.max_target_length, args.max_comment_length, args.num_train_epochs,
                         args.weightAB, args.weightBB, args.weightcon, args.autonum, dev_bleu, xmatch, best_bleu])
                    fcsv.close()


if __name__ == "__main__":
    main()

