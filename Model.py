import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset

from transformers import (RobertaTokenizer, T5Config, T5ForConditionalGeneration, 
                          AdamW, get_linear_schedule_with_warmup)
import numpy as np
from datasets import load_dataset

import random
import math
import json
from _utils import *
import os
from tqdm import tqdm
class Model_Class ():
    def __init__(self, config):
        self.__dict__ = config
        self.load_writers()
        self.load_model()
        self.train_filename = f"{self.data_dir}/{self.train_filename}"
        self.dev_filename = f"{self.data_dir}/{self.dev_filename}"
        self.test_filename = f"{self.data_dir}/{self.test_filename}"

    def convert_examples_to_features(self, item):
        example, example_index, tokenizer, args, stage = item

        source_str = "{}: {}".format(args.task, example.source)

        source_str = source_str.replace('</s>', '<unk>')
        source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
        assert source_ids.count(tokenizer.eos_token_id) == 1
        if stage == 'test':
            target_ids = []
        else:
            target_str = example.target
            target_str = target_str.replace('</s>', '<unk>')
            target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                        truncation=True)
            assert target_ids.count(tokenizer.eos_token_id) == 1

        return InputFeatures(
            example_index,
            source_ids,
            target_ids,
            url=example.url
        )
    
    def save_model_state(self, location):
        torch.save(self.model.state_dict(), f"{self.output_dir}/{location}")
    
    def training_loop(self, train_data = None,split_tags = None):
        # Load Train DataLoader
        if train_data == None:
            train_examples, train_data = self.load_data(filename  = self.train_filename, 
                                        split_tag = 'train' if split_tags is None else split_tags[0])
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, 
                                      sampler=train_sampler, 
                                      batch_size=self.train_batch_size,
                                      num_workers=4, 
                                      pin_memory=True)

        # Load Optimizer and Scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(self.learning_rate), eps=float(self.adam_epsilon))
        num_train_optimization_steps = self.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        # Log Info
        train_example_num = len(train_data)
        self.log_info(1, "***** Running training *****")
        self.log_info(1, "  Num examples = %d", train_example_num)
        self.log_info(1, "  Batch size = %d", self.train_batch_size)
        self.log_info(1, "  Batch num = %d", math.ceil(train_example_num / self.train_batch_size))
        self.log_info(1, "  Num epoch = %d", self.num_train_epochs)
        
        # Parameters
        self.dev_dataset = {}
        self.global_step, best_bleu_em, self.best_ppl = 0, -1, 1e6
        self.not_loss_dec_cnt, self.not_bleu_em_inc_cnt = 0, 0 if self.do_eval_bleu else 1e6

        # Main Training Loop
        for cur_epoch in range(self.start_epoch, int(self.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train(True)

            for step, batch in enumerate(bar):
                
                batch = tuple(t.to(self.device) for t in batch)
                
                source_ids, target_ids = batch
                source_mask = source_ids.ne(self.tokenizer.pad_token_id)
                target_mask = target_ids.ne(self.tokenizer.pad_token_id)
                
                self.model.config.decoder_start_token_id = self.decoder_start_token_id
                outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                     labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
                
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                
                if nb_tr_steps % self.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1
                    train_loss = round(tr_loss * self.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            if self.do_eval :
                self.eval(cur_epoch, split_tags=split_tags)
                if self.do_eval_bleu :
                    self.eval_bleu(cur_epoch)

    def eval_bleu(self, cur_epoch):
        pass

    def eval(self, cur_epoch, split_tags = None):
        # Eval model with dev dataset
        if 'dev_loss' in self.dev_dataset:
            eval_examples, eval_data = self.dev_dataset['dev_loss']
        else:            
            eval_examples, eval_data = self.load_data(filename=self.dev_filename, 
                                                      split_tag='dev' if split_tags is None else split_tags[1])
            self.dev_dataset['dev_loss'] = eval_examples, eval_data

        eval_ppl = self.eval_ppl_epoch(eval_data, eval_examples)
        result = {'epoch': cur_epoch, 'global_step': self.global_step, 'eval_ppl': eval_ppl}
        for key in sorted(result.keys()):
            self.log_info(1, "  %s = %s", key, str(result[key]))
        self.log_info(1, "  " + "*" * 20)
        # if self.data_num == -1:
            # tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

        if eval_ppl < self.best_ppl:
            self.not_loss_dec_cnt = 0
            self.log_info(1, "  Best ppl:%s", eval_ppl)
            self.log_info(1, "  " + "*" * 20)
            # fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
            self.best_ppl = eval_ppl

            # Save best checkpoint for best ppl
            output_dir = os.path.join(self.output_dir, 'checkpoint-best-ppl')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if self.always_save_model:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                self.log_info(1, "Save the best ppl model into %s", output_model_file)
        else:
            self.not_loss_dec_cnt += 1
            self.log_info(1, "Ppl does not decrease for %d epochs",self.not_loss_dec_cnt)
            if all([x > self.patience for x in [ self.not_bleu_em_inc_cnt,self.not_loss_dec_cnt]]):
                early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                    cur_epoch, self.not_bleu_em_inc_cnt, self.not_loss_dec_cnt)
                self.log_info(1, early_stop_str)
                # fa.write(early_stop_str)
                # return(break)
        self.log_info(1, "***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()

    def read_concode_examples(self, filename, data_num):
        """Read examples from filename."""
        examples = []

        with open(filename) as f:
            for idx, line in enumerate(f):
                x = json.loads(line)
                examples.append(
                    Example(
                        idx=idx,
                        source=x["nl"].strip(),
                        target=x["code"].strip()
                    )
                )
                idx += 1
                if idx == data_num:
                    break
        return examples

    def load_data(self, filename, split_tag, is_sample = False, only_src = False, cache_fn=None):
        data_tag = '_all' if self.data_num == -1 else '_%d' % self.data_num
        if cache_fn == None:
            cache_fn = '{}/{}.pt'.format(self.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
        examples = self.read_concode_examples(filename, self.data_num)

        if self.is_sample or is_sample:
            examples = random.sample(examples, min(self.sample_size, len(examples)))
        # if split_tag == 'train':
        #     calc_stats(examples, self.tokenizer, is_tokenize=True)
        # else:
        #     calc_stats(examples)
        if os.path.exists(cache_fn) and not self.is_sample:
            self.log_info(1, "Load cache data from %s", cache_fn)
            data = torch.load(cache_fn)
        else:
            if self.is_sample:
                self.log_info(1, "Sample 5k data for computing bleu from %s", filename)
            else:
                self.log_info(1, "Create cache data into %s", cache_fn)
            tuple_examples = [(example, idx, self.tokenizer, self, split_tag) for idx, example in enumerate(examples)]
            features = list(map(self.convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples))))
            all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
            if split_tag == 'test' or only_src:
                data = TensorDataset(all_source_ids)
            else:
                all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                data = TensorDataset(all_source_ids, all_target_ids)
            if not self.is_sample:
                torch.save(data, cache_fn)
        return examples, data

    def get_model_size(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model_size = sum([np.prod(p.size()) for p in model_parameters])
        return "{}M".format(round(model_size / 1e+6))

    def load_model(self, config = None, tokenizer = None, model = None, expirement = False, path = None):
        self.config = T5Config.from_pretrained(self.config_name) if config is None else config
        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name) if tokenizer is None else tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name) if model is None else model
        self.log_info(0,"Finish loading model [%s] from %s", self.get_model_size(), self.model_name)
        if self.load_model_path is not None and not expirement and path is None:
            self.log_info(0, "Reload model from {}".format(self.load_model_path))
            self.model.load_state_dict(torch.load(self.load_model_path))
        elif path is not None:
            self.log_info(0, "Reload model from {}".format(path))
            self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
    
    def load_writers(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.summary_fn = '{}/{}'.format(self.summary_dir, '/'.join(self.output_dir.split('/')[1:]))
        self.tb_writer = SummaryWriter(self.summary_fn)

    def log_info(self, level, information,*args):
        if (self.log_verbose and level <= self.info_level):
            self.logger.info(information,*args)

    def write_summary(self):
        if self.summary_verbose:
            print('ok') #fix

    def eval_ppl_epoch(self, eval_data, eval_examples):
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size,
                                    num_workers=4, pin_memory=True)
        # Start evaluating model
        self.log_info(1, "  " + "***** Running ppl evaluation *****")
        self.log_info(1, "  Num examples = %d", len(eval_examples))
        self.log_info(1, "  Batch size = %d", self.eval_batch_size)

        self.model.eval()
        eval_loss, batch_num = 0, 0
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, target_ids = batch
            source_mask = source_ids.ne(self.tokenizer.pad_token_id)
            target_mask = target_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

            eval_loss += loss.item()
            batch_num += 1
        eval_loss = eval_loss / batch_num
        eval_ppl = round(np.exp(eval_loss), 5)
        return eval_ppl

    def getter_function(self, attribute):
        return getattr(self,attribute)

    def setter_function(self, attribute, value):
        setattr(self,attribute,value)

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    def generate(self, input_sentence,min_length=20, max_length=50):
        input = self.tokenizer.encode(input_sentence, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input,min_length=min_length,max_length=max_length)
        return(self.tokenizer.decode(outputs[0], skip_special_tokens=True))