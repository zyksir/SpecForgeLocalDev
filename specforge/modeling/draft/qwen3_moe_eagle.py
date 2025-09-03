import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen3MoeConfig
from transformers.activations import ACT2FN
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRMSNorm, Qwen3MoeSparseMoeBlock, load_balancing_loss_func
from transformers.cache_utils import Cache
from .base import Eagle3DraftModel
from .llama3_eagle import LlamaRMSNorm, LlamaFlexAttention, LlamaAttention, prepare_decoder_attention_mask
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None
    print("grouped_gemm is not installed, please run `uv pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main`")

class SpecForgeQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_weight = torch.nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
        self.up_weight = torch.nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
        self.down_weight = torch.nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size))
        for i in range(config.num_experts):
            self.gate_weight.data[i].copy_(self.experts[i].gate_proj.weight.data)
            self.up_weight.data[i].copy_(self.experts[i].up_proj.weight.data)
            self.down_weight.data[i].copy_(self.experts[i].down_proj.weight.data)
        del self.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # [B, S, E]
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # [B, S, topk]
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        tokens_per_expert = selected_experts.view(-1).bincount(minlength=self.num_experts).to("cpu")
        permutated_hidden_states, row_id_map = grouped_gemm.ops.permute(hidden_states, selected_experts)
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gate_output = grouped_gemm.ops.gmm(permutated_hidden_states, self.gate_weight, tokens_per_expert, trans_b=True)
        gate_output = self.act_fn(gate_output)
        up_output = grouped_gemm.ops.gmm(permutated_hidden_states, self.up_weight, tokens_per_expert, trans_b=True)
        down_output = grouped_gemm.ops.gmm(gate_output * up_output, self.down_weight, tokens_per_expert, trans_b=True)
        final_hidden_states = grouped_gemm.ops.unpermute(down_output, row_id_map)
        return final_hidden_states, router_logits


class Qwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        router_logits = None
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


class Qwen3MoEForCausalLMEagle3(Eagle3DraftModel):
    config_class = Qwen3MoeConfig

    def __init__(self, config: Qwen3MoeConfig, quant_config=None, attention_backend="flex_attention", **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = Qwen3MoeDecoderLayer(config, layer_idx=0)
        q_norm = Qwen3MoeRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        k_norm = Qwen3MoeRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        def apply_qk_norm(q, k, q_norm, k_norm):
            original_q_shape = q.shape
            original_k_shape = k.shape
            q = q.view(-1, config.head_dim)
            k = k.view(-1, config.head_dim)
            q = q_norm(q)
            k = k_norm(k)
            q = q.view(original_q_shape)
            k = k.view(original_k_shape)
            return q, k
        if attention_backend == "flex_attention":
            self.midlayer.self_attn = LlamaFlexAttention(config=config, q_norm=q_norm, k_norm=k_norm)
            self.midlayer.self_attn.apply_qk_norm = apply_qk_norm
        elif attention_backend == "sdpa":
            self.midlayer.self_attn = LlamaAttention(config=config, q_norm=q_norm, k_norm=k_norm)
            self.midlayer.self_attn.apply_qk_norm = apply_qk_norm
        else:
            raise ValueError(f"Invalid attention backend: {attention_backend}")

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )
        # I don't want to change Qwen3MoeDecoderLayer so let's do it here
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
        **kwargs
    ):
        # The only difference is the Attention Part
        if ttt_length == 1:
            logger.info("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            logger.info(f"using ttt_length {ttt_length}, caching hidden states")
            cache_hidden = [[], []]
        
        batch_size, seq_length, _ = hidden_states.size()

        # make position ids
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        hidden_states = self.fc(hidden_states)

        hidden_states = self.hidden_norm(hidden_states)
        hidden_states, router_logits = self.midlayer(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,

            # For Custom Attention Part
            cache_hidden=cache_hidden,
            custom_hidden_states=hidden_states,

            # Other Arguments That might be used but we don't care
            cache_position=None,
            position_embeddings=None,
            output_attentions=False,
            output_router_logits=True,
            **kwargs
        ) # return MoeModelOutputWithPast
        hidden_states = self.norm(hidden_states)
        return hidden_states, router_logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
    
    def compute_router_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        # the range of router_logits is [topk, E], setting coef to 0.1 should be good.
        return self.config.router_aux_loss_coef * load_balancing_loss_func(router_logits, self.config.num_experts, self.config.num_experts_per_tok)
    
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.hidden_norm(hidden_states)
        return self.midlayer(
            hidden_states=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,

            # For Custom Attention Part
            cache_hidden=cache_hidden,
            custom_hidden_states=hidden_states,

            # Other Arguments That might be used but we don't care
            cache_position=None,
            position_embeddings=None,
            output_attentions=False,
            output_router_logits=True,
        )

def test_moe_block():
    from transformers import AutoConfig
    config = AutoConfig.from_file(".Qwen/Qwen3-30B-A3B-Instruct-2507")
    config.num_layers = 1
    base_model = Qwen3MoeSparseMoeBlock(config)
    spec_model = SpecForgeQwen3MoeSparseMoeBlock(config)
    spec_model.gate.weight.data.copy_(base_model.gate.weight.data)
    for i in range(config.num_experts):
        spec_model.gate_weight.data[i].copy_(base_model.experts[i].gate_proj.weight.data)
        spec_model.up_weight.data[i].copy_(base_model.experts[i].up_proj.weight.data)
        spec_model.down_weight.data[i].copy_(base_model.experts[i].down_proj.weight.data)

    # test forward
    spec_model = spec_model.to(torch.bfloat16).cuda()
    base_model = base_model.to(torch.bfloat16).cuda()
    hidden_states = torch.randn(1, 1024, config.hidden_size).to(torch.bfloat16).cuda()
    output_spec, router_logits_spec = spec_model(hidden_states)
    output_base, router_logits_base = base_model(hidden_states)
    assert torch.allclose(output_spec, output_base, atol=1e-4)
    assert torch.allclose(router_logits_spec, router_logits_base, atol=1e-4)

if __name__ == "__main__":
    test_moe_block()