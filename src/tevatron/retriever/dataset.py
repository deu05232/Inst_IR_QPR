import random
from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np

import re
from tevatron.retriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)
print_once = False


def format_query(query: str, prefix: str = '', prompt: str = '') -> str:
    global print_once
    if prompt.strip() != '':
        query_ends_in_punct = query.strip()[-1] in ['.', '?', '!']
        added_q = "" if query_ends_in_punct else "?"
        ret_str = f'{prefix} {query.strip()}{added_q} {prompt.strip()}'.strip()
    else:
        query_ends_in_punct = query.strip()[-1] in ['.', '?', '!']
        added_q = "" if query_ends_in_punct else "?"
        ret_str = f'{prefix} {query.strip()}{added_q}'.strip()

    if not print_once:
        logger.info(f'Prompt: `{prompt}`')
        logger.info(f'Query: {ret_str}')
        print_once = True

    return ret_str

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_n = min(self.data_args.negatives_first_n, len(group["new_negatives"]), negative_size)
            first_negs = group["new_negatives"][:first_n]
            remaining_to_select = negative_size - first_n
            # logger.info(f"first_n: {first_n}, remaining_to_select: {remaining_to_select}")
            
            if remaining_to_select > 0:
                _offset = epoch * remaining_to_select % len(group_negatives)
                shuffled_negs = [x for x in group_negatives]
                random.Random(_hashed_seed).shuffle(shuffled_negs)
                shuffled_negs = shuffled_negs * 2
                selected_negs = shuffled_negs[_offset: _offset + remaining_to_select]
            else:
                selected_negs = []

            negs = first_negs + selected_negs
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages
    
   
class FixedTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []
        

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages


class Idea3TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        
        self.length = len(self.train_data)
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        if group["has_instruction"]:
            only_query = group['only_query']
            only_instruction = group['only_instruction']
            formated_only_query = format_query(only_query, self.data_args.query_prefix, self.data_args.prompt)
            formated_query = formated_only_query + " " + only_instruction
        else:
            only_query = group['only_query']
            formated_only_query = format_query(only_query, self.data_args.query_prefix, self.data_args.prompt)
            formated_query = formated_only_query

        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        instruct_flag = 1 if group["has_instruction"] else 0
                
        return formated_query, formated_passages, formated_only_query, instruct_flag


class Idea3SplitedTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        
        self.length = len(self.train_data)
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        if group["has_instruction"]:
            only_query = group['only_query']
            only_instruction = group['only_instruction']
            formated_only_query = format_query(only_query, self.data_args.query_prefix, self.data_args.prompt)
            formated_query = formated_only_query + " instruction: " + only_instruction
        else:
            only_query = group['only_query']
            formated_only_query = format_query(only_query, self.data_args.query_prefix, self.data_args.prompt)
            formated_query = formated_only_query

        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        instruct_flag = 1 if group["has_instruction"] else 0
                
        return formated_query, formated_passages, formated_only_query, instruct_flag
    
    

class FixedTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []
        

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages


class FixedOriginalTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []
        

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages


def llama3_format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'<|start_header_id|>user<|end_header_id|>\n\n{prefix} {title.strip()} {text.strip()}'.strip()

def qwen25_format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'<|im_start|>user\n{prefix} {title.strip()} {text.strip()}'.strip()


class InstructTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None, tokenizer = None, model_name = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.model_name = model_name
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        if "llama" in self.model_name:
            formated_query = "<|start_header_id|>user<|end_header_id|>\n\n" + format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        elif "qwen2.5" in self.model_name:
            formated_query = "<|im_start|>user\n" + format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        else:
            raise ValueError(f"Model {self.model_name} is not supported for InstructTrainDataset.")
        
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        if "llama" in self.model_name:
            formated_passages.append(llama3_format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))
        elif "qwen2.5" in self.model_name:
            formated_passages.append(qwen25_format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))
        else:
            raise ValueError(f"Model {self.model_name} is not supported for InstructTrainDataset.")

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        if "llama" in self.model_name:
            for neg_psg in negs:
                formated_passages.append(llama3_format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))
        elif "qwen2.5" in self.model_name:
            for neg_psg in negs:
                formated_passages.append(qwen25_format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))
        else:
            raise ValueError(f"Model {self.model_name} is not supported for InstructTrainDataset.")

        return formated_query, formated_passages
    
    
class Idea3InstructTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None, tokenizer = None, model_name = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        self.tokenizer = tokenizer

        self.length = len(self.train_data)
        self.model_name = model_name
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        if group["has_instruction"]:
            only_query = group['only_query']
            only_instruction = group['only_instruction']
            formated_only_query = format_query(only_query, self.data_args.query_prefix, self.data_args.prompt)
            if "llama" in self.model_name:
                formated_query = "<|start_header_id|>user<|end_header_id|>\n\n" + formated_only_query + " " + only_instruction
            elif "qwen2.5" in self.model_name:
                formated_query = "<|im_start|>user\n" + formated_only_query + " " + only_instruction
            else:
                raise ValueError(f"Model {self.model_name} is not supported for Idea3InstructTrainDataset.")
        else:
            only_query = group['only_query']
            formated_only_query = format_query(only_query, self.data_args.query_prefix, self.data_args.prompt)
            if "llama" in self.model_name:
                formated_query = "<|start_header_id|>user<|end_header_id|>\n\n" + formated_only_query
            elif "qwen2.5" in self.model_name:
                formated_query = "<|im_start|>user\n" + formated_only_query
            else:
                raise ValueError(f"Model {self.model_name} is not supported for Idea3InstructTrainDataset.")
            

        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        if "llama" in self.model_name:
            formated_passages.append(llama3_format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))
        elif "qwen2.5" in self.model_name:
            formated_passages.append(qwen25_format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))
        else:
            raise ValueError(f"Model {self.model_name} is not supported for Idea3InstructTrainDataset.")

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        if "llama" in self.model_name:
            for neg_psg in negs:
                formated_passages.append(llama3_format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))
        elif "qwen2.5" in self.model_name:
            for neg_psg in negs:
                formated_passages.append(qwen25_format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))
        else:
            raise ValueError(f"Model {self.model_name} is not supported for Idea3InstructTrainDataset.")

        instruct_flag = 1 if group["has_instruction"] else 0
                
        return formated_query, formated_passages, formated_only_query, instruct_flag

    
class SplitedFixedTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        inst_flag = group["has_instruction"]
        if inst_flag:
            query = group['only_query'] + ' instruction: ' + group['only_instruction']
        else:
            query = group['query']
            
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix, self.data_args.prompt)
        formated_passages = []
        

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
            
            if remaining_to_select > 0:
                selected_negs = random.choices(group_negatives, k=remaining_to_select)
            else:
                selected_negs = []
                
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
            
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        elif self.data_args.negatives_first_n and self.data_args.negatives_first_n > 0:
            first_negs = group["d_inst_negatives"] + group['new_negatives']
            remaining_to_select = negative_size - len(first_negs)
                
            if remaining_to_select > 0:
                neg_passages = group['negative_passages']
                neg_passages = neg_passages * 2
                selected_negs = neg_passages[:remaining_to_select]
            else:
                selected_negs = []   
                       
            negs = first_negs + selected_negs
            assert len(negs) == negative_size
                
        else:
            assert False
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages
    

    
    
class EncodeDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = format_query(text['query'], self.data_args.query_prefix, self.data_args.prompt)
        else:
            text_id = text['docid']
            formated_text = format_passage(text['text'], text['title'], self.data_args.passage_prefix)
        return text_id, formated_text
