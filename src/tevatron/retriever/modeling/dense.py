import torch
import logging

from typing import Dict, Optional
from transformers import PreTrainedModel

from .encoder import EncoderModel, EncoderOutput
import torch.distributed as dist

from torch import nn, Tensor
from transformers import AutoTokenizer
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


        
class DenseModel_Idea3(EncoderModel):

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        
        
    def encode_query(self, qry, query_attn_mask=None, q_pooling_type="last"):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if query_attn_mask is not None:
            return self._pooling(query_hidden_states, qry['attention_mask']), self._pooling(query_hidden_states, qry['attention_mask'], query_attn_mask=query_attn_mask, q_pooling_type=q_pooling_type)
        else:
            return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
    
    def get_scores_q(self, query: Dict[str, Tensor], passage: Dict[str, Tensor]):  # , instruct_flag: Dict[int, Tensor]

        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
            
        scores = self.compute_similarity(q_reps, p_reps)
                    
        return scores


    def _pooling(self, last_hidden_state, attention_mask, query_attn_mask=None, q_pooling_type=""):
        if query_attn_mask is not None:
            if q_pooling_type == "last":
                sequence_lengths = query_attn_mask.sum(dim=1) - 2
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            elif q_pooling_type == "mean":
                masked_hiddens = last_hidden_state.masked_fill(~query_attn_mask[..., None].bool(), 0.0)
                reps = masked_hiddens.sum(dim=1) / query_attn_mask.sum(dim=1)[..., None]
            else:
                raise ValueError(f'unknown pooling method: {q_pooling_type}')
        elif self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, 
                instruct_flag: Dict[int, Tensor] = None, query_attn_mask: Dict[int, Tensor] = None, score_q = None):
         
        q_reps, only_q_reps = self.encode_query(query, query_attn_mask=query_attn_mask) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )
        
        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                only_q_reps = self._dist_gather_tensor(only_q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                instruct_flag = self._dist_gather_tensor(instruct_flag)
                
            # only_query_emb를 device에 올리고
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            # InfoNCE Loss
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            train_group_size = p_reps.size(0) // q_reps.size(0)
            target = target * train_group_size
            
            instruct_idx = torch.nonzero(instruct_flag, as_tuple=False).view(-1)
            instruct_target = target[instruct_idx]
            
            loss = self.compute_loss(scores / self.temperature, target)
            
            loss_kl_div = None
            if instruct_idx.numel() != 0:
                only_q_scores = self.compute_similarity(only_q_reps, p_reps)
                only_q_scores = only_q_scores.view(only_q_reps.size(0), -1)
                
                selected_only_q_scores = only_q_scores[instruct_idx]
                selected_score_q = score_q[instruct_idx]
                
                teacher_targets = torch.softmax(selected_score_q.detach(), dim=-1)
                loss_kl_div = - torch.mean(
                    torch.sum(torch.log_softmax(selected_only_q_scores, dim=-1) * teacher_targets, dim=-1))
                
            if self.process_rank == 0:
                print(f"loss:{loss}")
                if instruct_idx.numel() != 0:
                    print(f"kl_div:{loss_kl_div}")
                    print(f"selected_score_q: {selected_score_q[0,:16]}")
                    print(f"selected_only_q_scores: {selected_only_q_scores[0,:16]}")

            if loss_kl_div is not None:
                loss = loss + loss_kl_div

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
        

class DenseModel_Idea3_MSE(EncoderModel):

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        self.rank_loss = torch.nn.MSELoss()
        self.kl_loss_weight = 1.0
        self.kl_temp = temperature
        self.use_idx = 5
        
    def encode_query(self, qry, query_attn_mask=None, q_pooling_type="last"):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if query_attn_mask is not None:
            return self._pooling(query_hidden_states, qry['attention_mask']), self._pooling(query_hidden_states, qry['attention_mask'], query_attn_mask=query_attn_mask, q_pooling_type=q_pooling_type)
        else:
            return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
    
    def get_scores_q(self, query: Dict[str, Tensor], passage: Dict[str, Tensor]):  # , instruct_flag: Dict[int, Tensor]

        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
            
        scores = self.compute_similarity(q_reps, p_reps)
                    
        return scores


    def _pooling(self, last_hidden_state, attention_mask, query_attn_mask=None, q_pooling_type=""):
        if query_attn_mask is not None:
            if q_pooling_type == "last":
                sequence_lengths = query_attn_mask.sum(dim=1) - 2  # eos token이 없으므로
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            elif q_pooling_type == "mean":
                masked_hiddens = last_hidden_state.masked_fill(~query_attn_mask[..., None].bool(), 0.0)
                reps = masked_hiddens.sum(dim=1) / query_attn_mask.sum(dim=1)[..., None]
            else:
                raise ValueError(f'unknown pooling method: {q_pooling_type}')
        elif self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, 
                instruct_flag: Dict[int, Tensor] = None, query_attn_mask: Dict[int, Tensor] = None, score_q = None):
         
        q_reps, only_q_reps = self.encode_query(query, query_attn_mask=query_attn_mask) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )
        
        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                only_q_reps = self._dist_gather_tensor(only_q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                instruct_flag = self._dist_gather_tensor(instruct_flag)
                
            # only_query_emb를 device에 올리고
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            # InfoNCE Loss
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            train_group_size = p_reps.size(0) // q_reps.size(0)
            target = target * train_group_size
            
            instruct_idx = torch.nonzero(instruct_flag, as_tuple=False).view(-1)
            instruct_target = target[instruct_idx]
            
            loss = self.compute_loss(scores / self.temperature, target)
            
            loss_kl_div = None
            if instruct_idx.numel() != 0:
                only_q_scores = self.compute_similarity(only_q_reps, p_reps)
                only_q_scores = only_q_scores.view(only_q_reps.size(0), -1)
                
                teacher_pos = score_q[instruct_idx, instruct_target]
                teacher_neg1 = score_q[instruct_idx, instruct_target + self.use_idx]
                teacher_neg2 = score_q[instruct_idx, instruct_target + 6]
                teacher_neg3 = score_q[instruct_idx, instruct_target + 7]
                
                student_pos = only_q_scores[instruct_idx, instruct_target]
                student_neg1 = only_q_scores[instruct_idx, instruct_target + self.use_idx]
                student_neg2 = only_q_scores[instruct_idx, instruct_target + 6]
                student_neg3 = only_q_scores[instruct_idx, instruct_target + 7]
                
                student_margin1 = student_pos - student_neg1
                teacher_margin1 = teacher_pos - teacher_neg1
                student_margin2 = student_pos - student_neg2
                teacher_margin2 = teacher_pos - teacher_neg2
                student_margin3 = student_pos - student_neg3
                teacher_margin3 = teacher_pos - teacher_neg3
                
                loss_kl_div = (
                                self.rank_loss(student_margin1 / self.kl_temp, teacher_margin1 / self.kl_temp) +
                                self.rank_loss(student_margin2 / self.kl_temp, teacher_margin2 / self.kl_temp) +
                                self.rank_loss(student_margin3 / self.kl_temp, teacher_margin3 / self.kl_temp)
                            ) / 3
                
                loss_kl_div = torch.clamp(loss_kl_div, max=10.0)
                
            if self.process_rank == 0:
                print(f"loss:{loss}")
                if instruct_idx.numel() != 0:
                    print(f"kl_div:{loss_kl_div}")
                    print(f"q'_score: {student_pos[0]:.4f}, {student_neg1[0]:.4f}, {student_neg2[0]:.4f}, {student_neg3[0]:.4f}")
                    print(f"q_score: {teacher_pos[0]:.4f}, {teacher_neg1[0]:.4f}, {teacher_neg2[0]:.4f}, {teacher_neg3[0]:.4f}")
                    print(f"q_inst_score: {scores[instruct_idx[0], instruct_target[0]]:.4f}, {scores[instruct_idx[0], instruct_target[0] + self.use_idx]:.4f}, {scores[[instruct_idx[0], instruct_target[0] + 6]]:.4f}, {scores[[instruct_idx[0], instruct_target[0] + 7]]:.4f}")


            if loss_kl_div is not None:
                loss = loss + loss_kl_div * self.kl_loss_weight

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
        
    
        
class DenseModel_Idea3_MSE_rep(EncoderModel):
    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        self.rank_loss = torch.nn.MSELoss()
        
    def encode_query(self, qry, query_attn_mask=None, q_pooling_type="last"):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if query_attn_mask is not None:
            return self._pooling(query_hidden_states, qry['attention_mask']), self._pooling(query_hidden_states, qry['attention_mask'], query_attn_mask=query_attn_mask, q_pooling_type=q_pooling_type)
        else:
            return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
    
    def get_scores_q(self, query: Dict[str, Tensor], passage: Dict[str, Tensor]):  # , instruct_flag: Dict[int, Tensor]
        q_reps = self.encode_query(query)
        
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
                    
        return q_reps


    def _pooling(self, last_hidden_state, attention_mask, query_attn_mask=None, q_pooling_type=""):
        if query_attn_mask is not None:
            if q_pooling_type == "last":
                sequence_lengths = query_attn_mask.sum(dim=1) - 2
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            elif q_pooling_type == "mean":
                masked_hiddens = last_hidden_state.masked_fill(~query_attn_mask[..., None].bool(), 0.0)
                reps = masked_hiddens.sum(dim=1) / query_attn_mask.sum(dim=1)[..., None]
            else:
                raise ValueError(f'unknown pooling method: {q_pooling_type}')
        elif self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, 
                instruct_flag: Dict[int, Tensor] = None, query_attn_mask: Dict[int, Tensor] = None, score_q = None):
         
        q_reps, only_q_reps = self.encode_query(query, query_attn_mask=query_attn_mask) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )
        
        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                only_q_reps = self._dist_gather_tensor(only_q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                instruct_flag = self._dist_gather_tensor(instruct_flag)
                
            # only_query_emb를 device에 올리고
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            # InfoNCE Loss
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            train_group_size = p_reps.size(0) // q_reps.size(0)
            target = target * train_group_size
            
            instruct_idx = torch.nonzero(instruct_flag, as_tuple=False).view(-1)
            # instruct_target = target[instruct_idx]
            
            loss = self.compute_loss(scores / self.temperature, target)
            
            loss_kl_div = None
            if instruct_idx.numel() != 0:
                loss_kl_div = self.rank_loss(only_q_reps[instruct_idx], score_q[instruct_idx])
                # loss_kl_div = torch.clamp(loss_kl_div, max=10.0)
                
            if self.process_rank == 0:
                print(f"loss:{loss}")
                if instruct_idx.numel() != 0:
                    print(scores[instruct_idx[0], instruct_idx[0] * 16 : instruct_idx[0] * 16 + 4])
                    print(f"kl_div:{loss_kl_div}")
                    

            if loss_kl_div is not None:
                loss = loss + loss_kl_div

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
        
        
        
class DenseModel_Idea3_reduced_KL(EncoderModel):

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        
        
    def encode_query(self, qry, query_attn_mask=None, q_pooling_type="last"):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if query_attn_mask is not None:
            return self._pooling(query_hidden_states, qry['attention_mask']), self._pooling(query_hidden_states, qry['attention_mask'], query_attn_mask=query_attn_mask, q_pooling_type=q_pooling_type)
        else:
            return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
    
    def get_scores_q(self, query: Dict[str, Tensor], passage: Dict[str, Tensor]):  # , instruct_flag: Dict[int, Tensor]

        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
            
        scores = self.compute_similarity(q_reps, p_reps)
                    
        return scores


    def _pooling(self, last_hidden_state, attention_mask, query_attn_mask=None, q_pooling_type=""):
        if query_attn_mask is not None:
            if q_pooling_type == "last":
                sequence_lengths = query_attn_mask.sum(dim=1) - 2
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            elif q_pooling_type == "mean":
                masked_hiddens = last_hidden_state.masked_fill(~query_attn_mask[..., None].bool(), 0.0)
                reps = masked_hiddens.sum(dim=1) / query_attn_mask.sum(dim=1)[..., None]
            else:
                raise ValueError(f'unknown pooling method: {q_pooling_type}')
        elif self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, 
                instruct_flag: Dict[int, Tensor] = None, query_attn_mask: Dict[int, Tensor] = None, score_q = None):
         
        q_reps, only_q_reps = self.encode_query(query, query_attn_mask=query_attn_mask) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )
        
        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                only_q_reps = self._dist_gather_tensor(only_q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                instruct_flag = self._dist_gather_tensor(instruct_flag)
                
            # only_query_emb를 device에 올리고
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            # InfoNCE Loss
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            train_group_size = p_reps.size(0) // q_reps.size(0)
            target = target * train_group_size
            
            instruct_idx = torch.nonzero(instruct_flag, as_tuple=False).view(-1)
            instruct_target = target[instruct_idx]
            
            loss = self.compute_loss(scores / self.temperature, target)
            
            loss_kl_div = None
            if instruct_idx.numel() != 0:
                only_q_scores = self.compute_similarity(only_q_reps, p_reps)
                only_q_scores = only_q_scores.view(only_q_reps.size(0), -1)
                
                # negative가 너무 많은 것 같아 제대로 최적화가 안되는 것 같아 수를 줄임 -> train group에 대해서만 진행하자
                repeated_instruct_idx = instruct_idx.repeat_interleave(train_group_size)
                repeated_instruct_target = []
                for start in instruct_target:
                    for i in range(train_group_size):
                        repeated_instruct_target.append(start + i)
                
                selected_only_q_scores = only_q_scores[repeated_instruct_idx, repeated_instruct_target].view(instruct_idx.numel(), train_group_size)
                selected_score_q = score_q[repeated_instruct_idx, repeated_instruct_target].view(instruct_idx.numel(), train_group_size)
                
                teacher_targets = torch.softmax(selected_score_q.detach(), dim=-1)
                loss_kl_div = - torch.mean(
                    torch.sum(torch.log_softmax(selected_only_q_scores, dim=-1) * teacher_targets, dim=-1))
                
            if self.process_rank == 0:
                print(f"loss:{loss}")
                if instruct_idx.numel() != 0:
                    print(f"kl_div:{loss_kl_div}")
                    print(f"selected_score_q: {selected_score_q[0]}")
                    print(f"selected_only_q_scores: {selected_only_q_scores[0]}")
                    
            if loss_kl_div is not None:
                loss = loss + loss_kl_div

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
        
        
        
        

class DenseModel_Idea3_reduced_KL_only_inst(EncoderModel):

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        
        
    def encode_query(self, qry, query_attn_mask=None, q_pooling_type="last"):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if query_attn_mask is not None:
            return self._pooling(query_hidden_states, qry['attention_mask']), self._pooling(query_hidden_states, qry['attention_mask'], query_attn_mask=query_attn_mask, q_pooling_type=q_pooling_type)
        else:
            return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
    

    def get_scores_q(self, query: Dict[str, Tensor]):  # , instruct_flag: Dict[int, Tensor]

        q_reps = self.encode_query(query)

        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
                    
        return q_reps


    def _pooling(self, last_hidden_state, attention_mask, query_attn_mask=None, q_pooling_type=""):
        if query_attn_mask is not None:
            if q_pooling_type == "last":
                sequence_lengths = query_attn_mask.sum(dim=1) - 2
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            elif q_pooling_type == "mean":
                masked_hiddens = last_hidden_state.masked_fill(~query_attn_mask[..., None].bool(), 0.0)
                reps = masked_hiddens.sum(dim=1) / query_attn_mask.sum(dim=1)[..., None]
            else:
                raise ValueError(f'unknown pooling method: {q_pooling_type}')
        elif self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, 
                query_attn_mask: Dict[int, Tensor] = None, only_q_reps = None):
         
        q_reps, only_q_hat_reps = self.encode_query(query, query_attn_mask=query_attn_mask) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )
        
        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                only_q_hat_reps = self._dist_gather_tensor(only_q_hat_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                
            # only_query_emb를 device에 올리고
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            # InfoNCE Loss
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            train_group_size = p_reps.size(0) // q_reps.size(0)
            target = target * train_group_size
            
            loss = self.compute_loss(scores / self.temperature, target)
            
            loss_kl_div = None

            only_q_hat_scores = self.compute_similarity(only_q_hat_reps, p_reps)
            only_q_hat_scores = only_q_hat_scores.view(only_q_hat_reps.size(0), -1)
            
            q_scores = self.compute_similarity(only_q_reps, p_reps)
            q_scores = q_scores.view(only_q_reps.size(0), -1)

            # negative가 너무 많은 것 같아 제대로 최적화가 안되는 것 같아 수를 줄임 -> train group에 대해서만 진행하자
            instruct_idx = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)

            repeated_instruct_idx = instruct_idx.repeat_interleave(train_group_size)
            repeated_instruct_target = torch.arange(scores.size(0) * train_group_size, device=scores.device, dtype=torch.long)
            
            selected_only_q_hat_scores = only_q_hat_scores[repeated_instruct_idx, repeated_instruct_target].view(scores.size(0), train_group_size)
            selected_score_q = q_scores[repeated_instruct_idx, repeated_instruct_target].view(scores.size(0), train_group_size)
            
            teacher_targets = torch.softmax(selected_score_q.detach(), dim=-1)
            loss_kl_div = - torch.mean(
                torch.sum(torch.log_softmax(selected_only_q_hat_scores, dim=-1) * teacher_targets, dim=-1))
                
            if self.process_rank == 0:
                print(f"loss:{loss}")
                print(f"kl_div:{loss_kl_div}")
                print(f"selected_score_q: {selected_score_q[0]}")
                print(f"selected_only_q_scores: {selected_only_q_hat_scores[0]}")
                    
            if loss_kl_div is not None:
                loss = loss + loss_kl_div * 0.1

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    
class DenseModel_Idea3_MSE_rep_only_inst(EncoderModel):
    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        self.rank_loss = torch.nn.MSELoss()
        
    def encode_query(self, qry, query_attn_mask=None, q_pooling_type="last"):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        if query_attn_mask is not None:
            return self._pooling(query_hidden_states, qry['attention_mask']), self._pooling(query_hidden_states, qry['attention_mask'], query_attn_mask=query_attn_mask, q_pooling_type=q_pooling_type)
        else:
            return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
    
    def get_scores_q(self, query: Dict[str, Tensor]):  # , instruct_flag: Dict[int, Tensor]
        q_reps = self.encode_query(query)
        
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
                    
        return q_reps


    def _pooling(self, last_hidden_state, attention_mask, query_attn_mask=None, q_pooling_type=""):
        if query_attn_mask is not None:
            if q_pooling_type == "last":
                sequence_lengths = query_attn_mask.sum(dim=1) - 2
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            elif q_pooling_type == "mean":
                masked_hiddens = last_hidden_state.masked_fill(~query_attn_mask[..., None].bool(), 0.0)
                reps = masked_hiddens.sum(dim=1) / query_attn_mask.sum(dim=1)[..., None]
            else:
                raise ValueError(f'unknown pooling method: {q_pooling_type}')
        elif self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, 
                query_attn_mask: Dict[int, Tensor] = None, only_q_reps = None):
         
        q_reps, only_q_hat_reps = self.encode_query(query, query_attn_mask=query_attn_mask) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )
        
        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                only_q_hat_reps = self._dist_gather_tensor(only_q_hat_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            # only_query_emb를 device에 올리고
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            # InfoNCE Loss
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            train_group_size = p_reps.size(0) // q_reps.size(0)
            target = target * train_group_size
            
            loss = self.compute_loss(scores / self.temperature, target)
            loss_kl_div = self.rank_loss(only_q_hat_reps, only_q_reps)
                
            if self.process_rank == 0:
                print(f"loss:{loss}")
                print(f"kl_div:{loss_kl_div}")
                print(scores[0, 0:4])

            if loss_kl_div is not None:
                loss = loss + loss_kl_div

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
        