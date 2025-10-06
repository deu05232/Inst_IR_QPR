import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from modeling import EncoderModel
from transformers import AutoTokenizer


import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage  = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(TevatronTrainer, self)._get_train_sampler()




class WeightedTevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(WeightedTevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage, weight  = inputs
        return model(query=query, passage=passage, weight=weight).loss

    def training_step(self, *args):
        return super(WeightedTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(WeightedTevatronTrainer, self)._get_train_sampler()

class CurriculumPairwiseTevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(CurriculumPairwiseTevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage, instruct_flag  = inputs
        return model(query=query, passage=passage, pairwise_flag=instruct_flag).loss

    def training_step(self, *args):
        return super(CurriculumPairwiseTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(CurriculumPairwiseTevatronTrainer, self)._get_train_sampler()


class TevatronTrainer_w_Pairwise(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(TevatronTrainer_w_Pairwise, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"
        
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage, instruct_flag  = inputs
        # print(f"{query['device']} {self.tokenizer.batch_decode(query['input_ids'], skip_special_tokens=True)}")
        return model(query=query, passage=passage, instruct_flag=instruct_flag).loss  ##############

    def training_step(self, *args):
        return super(TevatronTrainer_w_Pairwise, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle:
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            return super(TevatronTrainer_w_Pairwise, self)._get_train_sampler()


class InstructFlagTevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(InstructFlagTevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage, instruct_flag  = inputs
        
        return model(query=query, passage=passage, instruct_flag=instruct_flag).loss

    def training_step(self, *args):
        return super(InstructFlagTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(InstructFlagTevatronTrainer, self)._get_train_sampler()


class DistanceWeightTevatronTrainerQinst(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(DistanceWeightTevatronTrainerQinst, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage, instruct_flag, alpha  = inputs
        
        return model(query=query, passage=passage, instruct_flag=instruct_flag, alpha=alpha).loss

    def training_step(self, *args):
        return super(DistanceWeightTevatronTrainerQinst, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(DistanceWeightTevatronTrainerQinst, self)._get_train_sampler()
        
        
class DistanceWeightTevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(DistanceWeightTevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage, only_query, instruct_flag, alpha  = inputs
                
        # query에 대한 weight를 계산
        with torch.no_grad():
            # only_query는 query와 동일해야함! 계산하기 어려워짐;;; -> 이부분은 추후 제거
            # assert query['input_ids'][0][0].shape == only_query['input_ids'][0][0].shape, f"{query['input_ids'].shape} {only_query['input_ids'].shape}"
            score_q = model.get_scores_q(query=only_query, passage=passage)  # , instruct_flag=instruct_flag
        return model(query=query, passage=passage, instruct_flag=instruct_flag, score_q=score_q, alpha=alpha).loss

    def training_step(self, *args):
        return super(DistanceWeightTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(DistanceWeightTevatronTrainer, self)._get_train_sampler()



class Idea3TevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(Idea3TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def compute_loss(self, model, inputs):
        query, passage, only_query, instruct_flag  = inputs
                
        # query에 대한 weight를 계산 -> query 부분도 계산하나보네..?
        with torch.no_grad():
            score_q = model.get_scores_q(query=only_query, passage=passage)  # , instruct_flag=instruct_flag
            
        return model(query=query, passage=passage, instruct_flag=instruct_flag, query_attn_mask=only_query["attention_mask"], score_q=score_q).loss


    def training_step(self, *args):
        return super(Idea3TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(Idea3TevatronTrainer, self)._get_train_sampler()


class Idea3ProgressiveTevatronTrainer(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(Idea3ProgressiveTevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def compute_loss(self, model, inputs):
        query, passage, only_query, instruct_flag, alpha  = inputs
                
        # query에 대한 weight를 계산
        with torch.no_grad():
            score_q = model.get_scores_q(query=only_query, passage=passage)  # , instruct_flag=instruct_flag
            
        return model(query=query, passage=passage, instruct_flag=instruct_flag, 
                     query_attn_mask=only_query["attention_mask"], score_q=score_q, alpha=alpha).loss


    def training_step(self, *args):
        return super(Idea3ProgressiveTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(Idea3ProgressiveTevatronTrainer, self)._get_train_sampler()
        
        
        
class Idea3TevatronTrainer_only_inst(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(Idea3TevatronTrainer_only_inst, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def compute_loss(self, model, inputs):
        query, passage, only_query, _  = inputs
                
        with torch.no_grad():
            only_q_reps = model.get_scores_q(query=only_query) 
        return model(query=query, passage=passage, query_attn_mask=only_query["attention_mask"], only_q_reps=only_q_reps).loss


    def training_step(self, *args):
        return super(Idea3TevatronTrainer_only_inst, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(Idea3TevatronTrainer_only_inst, self)._get_train_sampler()


class Idea3TevatronTrainer_only_inst_freeze(Trainer):
    def __init__(self, dont_shuffle, *args, **kwargs):
        super(Idea3TevatronTrainer_only_inst_freeze, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.dont_shuffle = "dont_shuffle" if dont_shuffle else "shuffle"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def compute_loss(self, model, inputs):
        query, passage, only_query, only_q_reps  = inputs

        only_q_reps = only_q_reps.to(model.device)

        return model(query=query, passage=passage, query_attn_mask=only_query["attention_mask"], only_q_reps=only_q_reps).loss


    def training_step(self, *args):
        return super(Idea3TevatronTrainer_only_inst_freeze, self).training_step(*args) / self._dist_loss_scale_factor

    def _get_train_sampler(self):
        if self.dont_shuffle == "dont_shuffle":
            print(f"***************Using dont_shuffle={self.dont_shuffle}")
            print(f"***************Using SequentialSampler for training")
            from torch.utils.data.sampler import SequentialSampler
            return SequentialSampler(self.train_dataset)
        else:
            print("Shuffling Dataset")
            return super(Idea3TevatronTrainer_only_inst_freeze, self)._get_train_sampler()
