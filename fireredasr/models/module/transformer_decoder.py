from typing import List, Optional, Dict
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import os
import torch
import torch.nn as nn
import xformers.ops as xops

from flash_attn import flash_attn_func, flash_attn_varlen_func

import einops

ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "SDPA") # Option: "NATIVE", "SDPA", "XFORMERS"
MultiHeadAttention = None
print("ATTENTION_BACKEND: ", ATTENTION_BACKEND)


class TransformerDecoder(nn.Module):
    def __init__(
            self, sos_id, eos_id, pad_id, odim,
            n_layers, n_head, d_model,
            residual_dropout=0.1, pe_maxlen=5000):
        super().__init__()
        self.INF = 1e10
        # parameters
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        # Components
        self.tgt_word_emb = nn.Embedding(odim, d_model, padding_idx=self.pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = DecoderLayer(d_model, n_head, residual_dropout)
            self.layer_stack.append(block)

        self.tgt_word_prj = nn.Linear(d_model, odim, bias=False)
        self.layer_norm_out = nn.LayerNorm(d_model)

        self.tgt_word_prj.weight = self.tgt_word_emb.weight
        self.scale = (d_model ** 0.5)
        self.scores_map = {}
        self.stride_map = {}
        self.filter_indexes = {}
        self.active_masks = {}
        
        
    def clear(self):
        for dec_layer in self.layer_stack:
            dec_layer.clear()        
        

    def batch_beam_search(self, encoder_outputs, src_masks,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        M = N * B
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(M, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(M, -1, Ti)
        ys = torch.ones(M, 1, device=device).fill_(self.sos_id).long()
        caches: List[Optional[Tensor]] = []
        for _ in range(self.n_layers):
            caches.append(torch.empty(M, 0, H, device=device, dtype=encoder_outputs.dtype))
            
        if B not in self.scores_map:
            scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
            self.scores_map[B] = scores
        scores = self.scores_map[B].repeat(N).view(M, 1)
        finished_mask_score = self.scores_map[B]
        is_finished = torch.zeros_like(scores)
        
        if (B, N) not in self.stride_map:
            stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(M).to(device).long()
            filter_index = torch.arange(M, device=device, dtype=torch.int32)
            active_mask = torch.ones(M, dtype=torch.bool, device=device)
            self.stride_map[(B, N)] = stride
            self.filter_indexes[M] = filter_index 
            self.active_masks[M] = active_mask
        stride = self.stride_map[(B, N)]
        active_mask = self.active_masks[M]
        active_indices = self.filter_indexes[M]
        last_t_logit = None

        # Autoregressive Prediction
        for t in range(maxlen):
            if ATTENTION_BACKEND == "FLASH_ATTN":
                tgt_mask = ys
            else:
                tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)
                
            dec_output = self.tgt_word_emb(ys) * self.scale + self.positional_encoding(ys)
            
            def expand(f, mask, indices, value=0.0, t=None):
                if t is None:
                    t = torch.full((len(mask), *list(f.shape[1:])), value, dtype=f.dtype, device=f.device)
                t[indices] = f
                return t

            dec_output = dec_output[active_indices]
            t_encoder_outputs = encoder_outputs[active_indices]
            tgt_mask = tgt_mask[active_indices]
            t_src_mask = src_mask[active_indices]
            
            for i, dec_layer in enumerate(self.layer_stack):
                dec_output = dec_layer.forward(
                    dec_output, 
                    t_encoder_outputs,
                    tgt_mask, 
                    t_src_mask,
                    cache=caches[i][active_indices],
                    active_indices=active_indices)
                caches[i] = dec_output          

            dec_output = self.layer_norm_out(dec_output)

            t_logit = self.tgt_word_prj(dec_output[:, -1])
            if last_t_logit is None:
                last_t_logit = t_logit
            else:
                last_t_logit[active_indices] = t_logit
            t_logit = last_t_logit            
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)
            
            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished, mask_score=finished_mask_score)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            # Accumulated
            scores = scores + t_topB_scores

            # Pruning 
            scores = scores.view(N, B*B)
            scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
            scores = scores.view(-1, 1)
        
            topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(M)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride

            # Update ys
            ys = ys[topB_row_number_in_ys]
            t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(M, 1)
            ys = torch.cat((ys, t_ys), dim=1)

            # Update caches
            new_caches: List[Optional[Tensor]] = []
            target = torch.full((len(active_mask), *list(caches[0].shape[1:])), 
                                        0.0, 
                                        dtype=caches[0].dtype, 
                                        device=caches[0].device)
            for cache in caches:
                new_caches.append(expand(cache, active_mask, active_indices,t=target)[topB_row_number_in_ys])
            caches = new_caches

            # Update finished state
            is_finished = t_ys.eq(self.eos_id)
            is_finished_n = is_finished.sum().item()
            active_mask = ~is_finished.squeeze()  
            #active_indices = self.filter_indexes[M][active_mask]
            active_indices = torch.nonzero_static(active_mask, size=M - int(is_finished_n)).squeeze(1)

            if is_finished_n == M:
                break
            
        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(M, -1)[index.view(-1)].view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(M)[index.view(-1)].view(N, -1)

        return [
            [
                {"yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]}
                for i, _ in enumerate(nbest_scores[n])
            ]
            for n in range(N)
        ]

    def ignored_target_position_is_0(self, padded_targets, ignore_id):
        mask = torch.ne(padded_targets, ignore_id)
        mask = mask.unsqueeze(dim=1)
        T = padded_targets.size(-1)
        upper_tri_0_mask = self.upper_triangular_is_0(T).unsqueeze(0).to(mask.dtype).to(mask.device)
        return mask.to(torch.uint8) & upper_tri_0_mask.to(torch.uint8)

    def upper_triangular_is_0(self, size):
        ones = torch.ones(size, size)
        tri_left_ones = torch.tril(ones)
        return tri_left_ones.to(torch.uint8)

    def set_finished_beam_score_to_zero(self, scores, is_finished, mask_score):
        NB, B = scores.size()
        is_finished = is_finished.float()
        mask_score = mask_score.view(1, B).repeat(NB, 1)
        return scores * (1 - is_finished) + mask_score * is_finished

    def set_finished_beam_y_to_eos(self, ys, is_finished):
        is_finished = is_finished.long()
        return ys * (1 - is_finished) + self.eos_id * is_finished

    def get_ys_lengths(self, ys):
        N, B, Tmax = ys.size()
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()



class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        if ATTENTION_BACKEND.upper() == "NATIVE":
            MultiHeadAttention = DecoderMultiHeadAttention
        elif ATTENTION_BACKEND.upper() == "SDPA":
            MultiHeadAttention = DecoderMHATorchSDPA
        elif ATTENTION_BACKEND.upper() == "XFORMERS":
            MultiHeadAttention = DecoderMHAXFormers
        elif ATTENTION_BACKEND.upper() == "FLASH_ATTN":
            MultiHeadAttention = DecoderMHAFlashAttn
        else:
            print("Unsupported attention backend: ", ATTENTION_BACKEND)
            exit(1)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model*4, dropout)
        
    def clear(self):
        self.cross_attn.clear()

    def forward(self, dec_input, enc_output, self_attn_mask, cross_attn_mask,
                cache=None, active_indices=None):
        x = dec_input
        residual = x
        x = self.self_attn_norm(x)
        if cache.shape[1]:
            xq = x[:, -1:, :]
            residual = residual[:, -1:, :]
            self_attn_mask = self.self_attn.parse_mask(self_attn_mask)
        else:
            xq = x
        x = residual + self.self_attn(xq, x, x, mask=self_attn_mask)
        residual = x
        x = residual+self.cross_attn(self.cross_attn_norm(x), 
                                     enc_output, 
                                     enc_output, 
                                     mask=cross_attn_mask, 
                                     is_cross=True,
                                     active_indices=active_indices)

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        x = torch.cat([cache, x], dim=1)
        return x
    
class BaseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)
        self.fc = nn.Linear(n_head * self.d_k, d_model)

    def forward(self, q, k, v, mask=None, is_cross=False, active_indices=None):
        raise NotImplementedError
    
    def clear(self):
        pass
    
    def parse_mask(self, mask):
        return mask[:, -1:, :]

# Native MHA
class DecoderMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__(d_model, n_head, dropout)
        self.attention = DecoderScaledDotProductAttention(temperature=self.d_k ** 0.5)

    def forward(self, q, k, v, mask=None, is_cross=False, active_indices=None):
        bs = q.size(0)

        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.fc(output)

        return output

class DecoderScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.INF = float("inf")

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output

# MHA with Torch SDPA
class DecoderMHATorchSDPA(BaseMultiHeadAttention):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__(d_model, n_head, dropout)
        self.attention = DecoderTorchSDPA(temperature=self.d_k ** 0.5)

    def forward(self, q, k, v, mask=None, is_cross=False, active_indices=None):
        bs = q.size(0)

        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)

        output = self.attention(q, k, v, mask=mask.unsqueeze(1))
        output = self.fc(output.transpose(1, 2).contiguous().view(bs, -1, self.d_model))
        return output

class DecoderTorchSDPA(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch, num_heads, seq_len, d_k)
        mask: optional attention mask
              - If boolean: shape (batch, 1, seq_len, seq_len) or broadcastable.
                True means 'mask out'.
              - If float: same shape, with -inf for masked positions.
        """
        # F.scaled_dot_product_attention will:
        # - scale internally
        # - apply softmax
        # - apply mask if given
        # - compute attention output
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask.eq(1),
            dropout_p=0.0,          # set >0 only during training
            is_causal=False,        # set True to get causal masking automatically
        )
        return output

# MHA with xFormers
class DecoderMHAXFormers(BaseMultiHeadAttention):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__(d_model, n_head, dropout)

    def forward(self, q, k, v, mask=None, is_cross=False, active_indices=None):
        bs = q.size(0)

        # projection and transform to (batch*n_head, seq_len, head_dim)
        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k).transpose(1, 2).reshape(bs * self.n_head, -1, self.d_k)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k).transpose(1, 2).reshape(bs * self.n_head, -1, self.d_k)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k).transpose(1, 2).reshape(bs * self.n_head, -1, self.d_k)
        # single-step reshape+transpose
        #q = einops.rearrange(self.w_qs(q), 'b s (h d) -> (b h) s d', h=self.n_head)
        #k = einops.rearrange(self.w_ks(k), 'b s (h d) -> (b h) s d', h=self.n_head)
        #v = einops.rearrange(self.w_vs(v), 'b s (h d) -> (b h) s d', h=self.n_head)
        output = xops.memory_efficient_attention(q, k, v)
        # back to (bs, seq_len, d_model)
        output = output.reshape(bs, self.n_head, -1, self.d_k).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        #output = einops.rearrange(output, '(b h) s d -> b s (h d)', b=bs, h=self.n_head)
        output = self.fc(output)
        return output
    
    
class DecoderMHAFlashAttn(BaseMultiHeadAttention):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__(d_model, n_head, dropout)
        
        self.cross_cache_k = None
        self.cross_cache_v = None
        self.cross_cache_seqs = None
        
    def clear(self):
        self.cross_cache_k = None
        self.cross_cache_v = None
        self.cross_cache_seqs = None

    def forward(self, q, k, v, mask=None, is_cross=False, active_indices=None):
        is_casual = not is_cross
        bs = q.size(0)
        
        if is_casual:
            q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k)
            k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
            v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)
            output = flash_attn_func(q, k, v, causal=is_casual)
            output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        else:
            mask = mask.squeeze(1)
            bool_mask = mask.view(-1).bool()
            q = self.w_qs(q).view(-1, self.n_head, self.d_k)
            
            if self.cross_cache_k is None or self.cross_cache_v is None:
                k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
                v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)  
                
                seq_lens = mask.sum(dim=1, dtype=torch.int32)
                self.cross_cache_k = k
                self.cross_cache_v = v
                self.cross_cache_seqs = seq_lens
                
      
            seq_lens = self.cross_cache_seqs[active_indices]
            total, _max_seq_k = seq_lens.sum().item(), seq_lens.max().item()
            bool_indices = torch.nonzero_static(bool_mask, size=total).squeeze(1)
            var_k = self.cross_cache_k[active_indices].view(-1, self.n_head, self.d_k)[bool_indices]
            var_v = self.cross_cache_v[active_indices].view(-1, self.n_head, self.d_k)[bool_indices]
            cu_seqlens_q = torch.arange(0, bs + 1, 1, device=seq_lens.device, dtype=torch.int32)
            cu_seqlens_k = torch.zeros(bs + 1, device=seq_lens.device, dtype=torch.int32)
            cu_seqlens_k[1:] = torch.cumsum(seq_lens, dim=0)
            output = flash_attn_varlen_func(q=q, 
                                            k=var_k, 
                                            v=var_v, 
                                            cu_seqlens_q=cu_seqlens_q, 
                                            cu_seqlens_k=cu_seqlens_k, 
                                            max_seqlen_q=1, 
                                            max_seqlen_k=_max_seq_k, 
                                            causal=is_casual)
            output = output.contiguous().view(bs, -1, self.d_model)
        output = self.fc(output)
        return output
    
    def parse_mask(self, mask):
        return mask


# @torch.compile(mode="reduce-overhead", backend="inductor")
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        output = self.w_2(self.act(self.w_1(x)))
        return output

# @torch.compile(mode="reduce-overhead", backend="inductor")
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)
        return self.pe[:, :length].clone().detach()
