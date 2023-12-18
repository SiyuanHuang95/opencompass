from typing import Dict, List, Optional, Union, Any
from types import TracebackType
import inspect
import json
from types import SimpleNamespace 

import torch
import warnings
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.utils.components import RMSNorm
from opencompass.utils.tensor_type import default_tensor_type
from opencompass.models.sphinx_meta import MetaModel
from opencompass.utils.llama import precompute_freqs_cis, apply_rotary_emb, repeat_kv


import open_clip
from flash_attn import flash_attn_func
import functools
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass, field
import math
default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))
PromptType = Union[PromptList, str]

from fairscale.nn.model_parallel import initialize as fs_init
from opencompass.utils import misc, tensor_parallel
from opencompass.models.sphinx_tokenizer import Tokenizer

from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region
)

from opencompass.models.peft import LoraColumnParallelLinear, LoraRowParallelLinear, LoraLinear


class MixtralMoESphinx(BaseModel):
    """Mixtral model wrapper
    
    """
    def __init__(
        self,
        path: str,
        quant_flag: bool = False,
        max_seq_len: int = 2048,
        max_batch_size: int = 8,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
        num_gpus: int = 2,
    ):
        
        self._load_model(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_path=tokenizer_path,
            num_gpus = num_gpus,
            max_batch_size=max_batch_size,
            quant_flag=quant_flag,
        )
        
        self.max_seq_len = max_seq_len
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()
        self.params = {
           "max_batch_size": max_batch_size,
           "max_seq_len": max_seq_len,
        }
    
    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None,
                    num_gpus: int = 2,
                    quant_flag: bool = False):
        
        self.llama_type = "sphinx_moe"
        self.llama_config = ["/mnt/petrelfs/share_data/gaopeng/mixtral-8x7b-32kseqlen/ori/params.json"]
        tokenizer_path = "/mnt/petrelfs/share_data/gaopeng/mixtral-8x7b-32kseqlen/tokenizer.model"
        
        distributed_args = SimpleNamespace(
            model_parallel_size=2,
            # world_size=1,
            # local_rank=1,
            dist_url='env://'
        )
        
        misc.init_distributed_mode(distributed_args)
        fs_init.initialize_model_parallel(num_gpus)

        target_dtype = torch.bfloat16
        print("mp_group: ", {dist.get_world_size(fs_init.get_model_parallel_group())})
        self.model = MetaModel.from_pretrained(path, "mixtral_moe", self.llama_config, tokenizer_path,
                                    with_visual=False, max_seq_len=max_seq_len,
                                    mp_group=fs_init.get_model_parallel_group(),
                                    dtype=target_dtype, device="cuda",)
        
        # self.model = MetaModel(self.llama_type, self.llama_config, tokenizer_path, with_visual=False)
        # tensor_parallel.load_tensor_parallel_model_list(self.model, path)
        if quant_flag:
            print("Quantizing model to 4 bit!")
            from opencompass.utils.quant import quantize
            from transformers.utils.quantization_config import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig.from_dict(
                config_dict={
                    "load_in_8bit": False, 
                    "load_in_4bit": True, 
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                return_unused_kwargs=False,
            )
            quantize(self.model, quantization_config)
        
        self.model.bfloat16().cuda()
        self.tokenizer = self.model.tokenizer
        
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        assert mask_length is None, 'mask_length is not supported'

        # # tokenize
        # prompt_tokens = [self.tokenizer.encode(x, True, False) for x in inputs]
        # max_prompt_size = max([len(t) for t in prompt_tokens])
        
        # max_seq_len = self.params["max_seq_len"]
        # total_len = min(max_seq_len, max_prompt_size)
        # tokens = torch.zeros((bsz, total_len)).cuda().long()
        # for k, t in enumerate(prompt_tokens):
        #     num_token = min(total_len, len(t))
        #     tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
        # # forward
        # outputs = self.model.forward(tokens, 0)
        
        # # compute ppl
        # shift_logits = outputs[..., :-1, :].contiguous().float()
        bsz = len(inputs)
        max_batch_size = self.params["max_batch_size"]
        assert bsz <= max_batch_size, (bsz, max_batch_size)
        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in inputs]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        max_seq_len = self.params["max_seq_len"]
        total_len = min(max_seq_len, max_prompt_size)
        tokens = torch.full((bsz, total_len), 0).cuda().long()
        for k, t in enumerate(prompt_tokens):
            num_token = min(total_len, len(t))
            tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
        
        start_pos = min_prompt_size
        prev_pos = 0
        outputs = self.model.llma.forward_ppl(tokens, prev_pos)
        shift_logits = outputs[..., :-1, :].contiguous().float()
        
        shift_labels = tokens[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits, shift_labels).view(bsz, -1)
        lens = (tokens != 0).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss 
    
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
            
        results = self.model.generate(prompts=inputs, 
                                      images=None, 
                                      max_gen_len=max_out_len,
                                      temperature=0)
        return results
    
    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, True, True))
    



# try:
#     from apex.normalization import FusedRMSNorm as RMSNorm
# except ImportError:
#     warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")

#     class RMSNorm(torch.nn.Module):
#         def __init__(self, dim: int, eps: float = 1e-6):
#             """
#             Initialize the RMSNorm normalization layer.

#             Args:
#                 dim (int): The dimension of the input tensor.
#                 eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

#             Attributes:
#                 eps (float): A small value added to the denominator for numerical stability.
#                 weight (nn.Parameter): Learnable scaling parameter.

#             """
#             super().__init__()
#             self.eps = eps
#             self.weight = nn.Parameter(torch.ones(dim))

#         def _norm(self, x):
#             """
#             Apply the RMSNorm normalization to the input tensor.

#             Args:
#                 x (torch.Tensor): The input tensor.

#             Returns:
#                 torch.Tensor: The normalized tensor.

#             """
#             return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#         def forward(self, x):
#             """
#             Forward pass through the RMSNorm layer.

#             Args:
#                 x (torch.Tensor): The input tensor.

#             Returns:
#                 torch.Tensor: The output tensor after applying RMSNorm.

#             """
#             output = self._norm(x.float()).type_as(x)
#             return output * self.weight


# class default_tensor_type:
#     _tensor_type_stack = [(torch.float, "cpu")]
    
#     def __init__(
#         self,
#         dtype: Optional[torch.dtype] = None,
#         device: Optional[str] = None,
#     ) -> None:
#         # Only limited combinations are supported.
#         assert device is None or device in ["cpu", "cuda"]
#         assert dtype is None or dtype in [torch.float, torch.bfloat16, torch.half]
#         self.dtype, self.device = dtype, device
    
#     def __enter__(self) -> None:
#         dtype, device = self.dtype, self.device
#         if dtype is None:
#             dtype = default_tensor_type._tensor_type_stack[-1][0]
#         if device is None:
#             device = default_tensor_type._tensor_type_stack[-1][1]
#         default_tensor_type._tensor_type_stack.append((dtype, device))
        
#         # We use all 3 calls since the new apis (set_default_device, set_default_dtype)
#         # seems to be ineffective sometimes (e.g., set_default_device is ineffective to
#         # torch.Tensor calls).
#         torch.set_default_tensor_type(default_tensor_type.get_tensor_type(dtype, device))
#         torch.set_default_device(device)
#         torch.set_default_dtype(dtype)

#     def __exit__(
#         self,
#         exc_type: Optional[type[BaseException]],
#         exc_val: Optional[BaseException],
#         exc_tb: Optional[TracebackType],
#     ) -> None:
#         default_tensor_type._tensor_type_stack.pop()
#         dtype, device = default_tensor_type._tensor_type_stack[-1]

#         torch.set_default_tensor_type(default_tensor_type.get_tensor_type(dtype, device))
#         torch.set_default_device(device)
#         torch.set_default_dtype(dtype)

#     @staticmethod
#     def get_tensor_type(dtype: torch.dtype, device: str) -> Any:
#         return {
#             (torch.float, "cpu"): torch.FloatTensor,
#             (torch.bfloat16, "cpu"): torch.BFloat16Tensor,
#             (torch.half, "cpu"): torch.HalfTensor,
#             (torch.float, "cuda"): torch.cuda.FloatTensor,
#             (torch.bfloat16, "cuda"): torch.cuda.BFloat16Tensor,
#             (torch.half, "cuda"): torch.cuda.HalfTensor,
#         }[(dtype, device)]

@dataclass
class ModelArgs:
    dim: int = 4096
    hidden_dim: int = 16384
    head_dim: int = 128
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    norm_eps: float = 1e-5
    rope_theta: float = 1000000 # todo 1e6 really?

    max_batch_size: int = 32
    max_seq_len: int = 2048

    moe: Dict[str, int] = field(default_factory=lambda: {
        "num_experts_per_tok": 2,
        "num_experts": 8
    })
    load_balancing_weight: float = 0.1

    rope_scaling: Optional[float] = None

    lora_rank: int = 16 # lora
    bias_tuning: bool = True  # bias
    norm_tuning: bool = False


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = LoraColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wk = LoraColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wv = LoraColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wo = LoraRowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.bias_tuning,
            input_is_parallel=True,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )

        self.args = args

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        """
        Supported mask spec:

        1. Float tensor: The tensor is added to the attention score matrix.
        2. Boolean tensor: Substitute the ``True`` values with ``0.0`` and ``False`` values with 
           ``-inf``, then process in the same way as the float tensor.
        3. str: Currently the only supported choice is ``causal``, for which each token attends
           to all tokens appearing no later than itself. Our implementation assumes the query and
           key sequences aligns on the right for ``causal`` if their lengths are not equal.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # if cache is enabled, prepend keys and values in the history.
        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        is_causal = isinstance(mask, str) and mask == "causal"
        # "causal" dispatches to flash_attn only when q and k have the same seqlen
        # because currently the flash_attn causal impl for unequal q & k length is not suited
        # for generation: Generation with cache requires aligning on the right, while the
        # current flash_attn impl aligns on the left. For example, we expect the mask to be
        # as the left one, while the current flash_attn impl gives the right one
        #
        #              K                     K
        #        1 1 1 1 1 0 0         1 0 0 0 0 0 0
        #     Q  1 1 1 1 1 1 0       Q 1 1 0 0 0 0 0
        #        1 1 1 1 1 1 1         1 1 1 0 0 0 0
        use_flash = (
            self.flash  # user configuration
            and (mask is None or (is_causal and keys.size(1) == xq.size(1)))  # supported mask
        )
        if use_flash:
            # repeating k/v heads is included in flash_attn
            output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=is_causal)
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            if isinstance(mask, str):
                if is_causal:
                    mask = self._make_causal_mask(xq.size(2), keys.size(2))
                    mask = mask.to(xq.device, non_blocking=True)
                else:
                    raise NotImplementedError()
            output = F.scaled_dot_product_attention(xq, keys, values, dropout_p=0.0, attn_mask=mask)
            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None

    def _make_causal_mask(self, q_len: int, kv_len: int) -> torch.Tensor:
        q_indices = torch.arange(q_len) - q_len
        kv_indices = torch.arange(kv_len) - kv_len
        causal_mask_bool = q_indices.view(-1, 1) >= kv_indices.view(1, -1)
        return causal_mask_bool

class ExpertFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        args:ModelArgs
    ):
        super().__init__()
        self.w1 = LoraLinear(
            dim, hidden_dim, bias=args.bias_tuning, lora_rank=args.lora_rank
        )
        self.w2 = LoraLinear(
            hidden_dim, dim, bias=args.bias_tuning, lora_rank=args.lora_rank
        )
        self.w3 = LoraLinear(
            dim, hidden_dim, bias=args.bias_tuning, lora_rank=args.lora_rank
        )

        for param in self.parameters():
            # mark as model parallel parameters,
            # otherwise the params will be broadcast within model parallel group to ensure consistency among ranks
            param.is_model_parallel = True

    # @torch.compile
    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class MoE(nn.Module):
    LOAD_BALANCING_LOSSES = []
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        load_balancing_weight: float,
        args: ModelArgs
    ):
        super().__init__()
        mp_size = fs_init.get_model_parallel_world_size()
        mp_rank = fs_init.get_model_parallel_rank()
        assert num_experts % mp_size == 0
        n_local_experts = num_experts // mp_size
        self.num_experts = num_experts
        self.local_experts = [str(i) for i in range(n_local_experts*mp_rank, n_local_experts*(mp_rank+1))]
        self.experts = nn.ModuleDict({
            i : ExpertFeedForward(dim, hidden_dim, args) for i in self.local_experts
        })
        self.gate = nn.Linear(dim, num_experts, bias=False)
        def gate_grad_hook(grad):
            grad = grad.clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=fs_init.get_model_parallel_group())
            return grad
        for parameter in self.gate.parameters():
            parameter.register_hook(gate_grad_hook)

        self.num_experts_per_tok = num_experts_per_tok
        self.load_balancing_weight = load_balancing_weight

    def _load_balancing_loss(self, expert_scores, flat_expert_indices):
        """

        Args:
            expert_scores: size(n_tokens, num_experts), last dim sum to 1
            flat_expert_indices: size(n_tokens * num_experts_per_tok)

        Returns:

        """
        n_tokens = expert_scores.shape[0]
        # tokens_per_expert.shape == (num_experts)
        tokens_per_expert = torch.bincount(flat_expert_indices, minlength=self.num_experts).to(expert_scores)
        assert not tokens_per_expert.requires_grad
        scores = expert_scores.mean(dim=0)
        scale = (self.load_balancing_weight * self.num_experts) / (n_tokens * self.num_experts_per_tok)
        loss = scale * torch.dot(tokens_per_expert, scores)
        if fs_init.get_model_parallel_rank() != 0:
            loss = loss * 0  # gradient come from rank0 through all reduce
        return loss


    def forward(self, x):
        x = copy_to_model_parallel_region(x)
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x)
        scores = scores.softmax(dim=-1).to(x)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        flat_expert_indices = expert_indices.view(-1)

        if self.training:
            MoE.LOAD_BALANCING_LOSSES.append(self._load_balancing_loss(scores, flat_expert_indices))

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.zeros_like(x)
        for str_i, expert in self.experts.items():
            y[flat_expert_indices == int(str_i)] = expert(x[flat_expert_indices == int(str_i)])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)

        y = reduce_from_model_parallel_region(y)
        return y.view(*orig_shape).to(x)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MoE(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_experts=args.moe['num_experts'],
            num_experts_per_tok=args.moe["num_experts_per_tok"],
            load_balancing_weight=args.load_balancing_weight,
            args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask):
        return x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        h = self._forward_attention(x, start_pos, freqs_cis, mask)
        out = self._forward_ffn(h)
        return out


class Transformer(nn.Module):
    is_peft = True
    def __init__(self, args: ModelArgs, with_visual=False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=default_linear_init
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=default_linear_init
        )

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2,
            theta=self.args.rope_theta, scaling=self.args.rope_scaling
        )

        self.image_words = 0
        self.cache_image_words = 0 # for inference
        if with_visual:
            print("build llama model with clip")
            with default_tensor_type(dtype=torch.half):
                self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            in_dim = self.clip.visual.proj.shape[1]
            # in_dim = 3
            self.clip_proj = nn.Linear(in_dim, args.dim)
            self.clip_proj_norm = nn.LayerNorm(args.dim)
            self.image_words = 257


    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("clip."):
                trainable_key_words = []
                if self.args.lora_rank > 0:
                    trainable_key_words.append("lora")
                if self.args.norm_tuning:
                    trainable_key_words.append("norm")
                if self.args.bias_tuning:
                    trainable_key_words.append("bias")
                if any([_ in name for _ in trainable_key_words]):
                    trainable[name] = para

        return trainable


    @torch.no_grad()
    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x


    def encode_image(self, image):
        with torch.cuda.amp.autocast(enabled=False):
            image = image.half()
            image_tokens = self.clip_encode_image(image)
            image = image.to(self.clip_proj.weight.dtype)
        image_tokens = self.clip_proj_norm(self.clip_proj(image_tokens))
        return image_tokens


    def forward(self, examples, image=None):
        self._destroy_kv_cache()  # training always disables kv cache
        MoE.LOAD_BALANCING_LOSSES.clear()

        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        image_words = 0
        if image is not None:
            image_tokens = self.encode_image(image)
            image_words = image_tokens.shape[1]
            h = torch.cat((image_tokens, h), dim=1)
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        h = self.norm(h)
        output = self.output(h[:, image_words:, :])

        load_balancing_loss = sum(MoE.LOAD_BALANCING_LOSSES) / len(MoE.LOAD_BALANCING_LOSSES)
        return output, {"load_balancing": load_balancing_loss}


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            self._allocate_kv_cache(_bsz)  # kv cache will not re-allocate if size is unchanged
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            assert start_pos == 0
            image_tokens = self.encode_image(image)
            self.cache_image_words = image_tokens.shape[1]
            h = torch.cat((image_tokens, h), dim=1)
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
        else:
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # Despite that "causal" also works for seqlen == 1, keep it to None for possibly
        # better performance
        mask = None if seqlen == 1 else "causal"

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
    
    @torch.inference_mode()
    def forward_ppl(self, tokens, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        
        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(max_batch_size, self.args.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()