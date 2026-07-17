import torch
from transformers import Concatenate, WeightConverter
from transformers.conversion_mapping import register_checkpoint_conversion_mapping

# Register weight conversion for fused QKV attention
register_checkpoint_conversion_mapping(
    model_type="llama",  # or the specific model type you're using
    mapping=[
        WeightConverter(
            source_patterns=["q_proj", "k_proj", "v_proj"],
            target_patterns=["qkv_proj"],
            operations=[Concatenate(dim=0)],  # Concatenate weights along dimension 0
        ),
    ],
    overwrite=True,
)

print("Weight conversion mapping registered.")