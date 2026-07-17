import torch
import torch.nn as nn
from typing import Optional, Tuple

class PhiFusedQKVAttention(nn.Module):
    """
    Fused QKV attention for Phi models.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        
        # Get Phi-specific config values
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        # Fused QKV projection - combines q_proj, k_proj, v_proj
        total_dim = 3 * self.hidden_size
        self.qkv_proj = nn.Linear(self.hidden_size, total_dim, bias=False)
        
        # Output projection (called 'dense' in Phi)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning exactly 2 values for Phi compatibility"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Fused projection
        qkv = self.qkv_proj(hidden_states)
        
        # Split into Q, K, V
        query, key, value = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scale = self.head_dim ** 0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / scale
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.dense(attn_output)
        
        # Return exactly 2 values as Phi expects
        return attn_output, attn_weights