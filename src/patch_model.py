from transformers import AutoModelForCausalLM
from transformers.monkey_patching import register_patch_mapping, clear_patch_mapping
from src.custom_attention import FusedQKVAttention

# Register the patch globally
# This affects ALL subsequent model loads via from_pretrained() [citation:4]
register_patch_mapping(
    mapping={
        # Map the original LlamaAttention class to your custom version
        "LlamaAttention": FusedQKVAttention,
        # You can use regex patterns too: ".*Attention": FusedQKVAttention
    },
    overwrite=True  # Overwrite any existing patches for these classes
)

print("Monkey patch registered. Loading model...")

# Load a model - the patch is automatically applied during initialization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    device_map="auto"
)

# Verify the patch was applied
print(f"Attention layer type: {type(model.model.layers[0].self_attn)}")
# Should print: <class '__main__.FusedQKVAttention'>

# Clean up when done (important for long-running applications) [citation:4]
clear_patch_mapping()