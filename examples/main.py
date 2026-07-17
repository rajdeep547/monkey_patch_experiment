import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.monkey_patching import register_patch_mapping, clear_patch_mapping
from src.custom_attention import PhiFusedQKVAttention
import copy

def apply_monkey_patch():
    """Apply the monkey patch for Phi attention layers"""
    try:
        register_patch_mapping(
            mapping={
                "PhiAttention": PhiFusedQKVAttention,
            },
            overwrite=True
        )
        print("✓ Monkey patch registered successfully!")
        return True
    except Exception as e:
        print(f"⚠ Failed to register monkey patch: {e}")
        return False

def patch_with_weight_transfer(model):
    """
    Patch model and transfer weights by capturing them before replacement
    """
    patched_count = 0
    transferred_count = 0
    
    try:
        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            print("⚠ Could not find layers in model")
            return 0, 0
        
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                old_attn = layer.self_attn
                old_type = type(old_attn).__name__
                
                # Capture weights BEFORE creating new attention
                q_weight = None
                k_weight = None
                v_weight = None
                dense_weight = None
                dense_bias = None
                
                # Try to get weights from old attention
                if hasattr(old_attn, 'q_proj') and old_attn.q_proj is not None:
                    q_weight = old_attn.q_proj.weight.data.clone()
                if hasattr(old_attn, 'k_proj') and old_attn.k_proj is not None:
                    k_weight = old_attn.k_proj.weight.data.clone()
                if hasattr(old_attn, 'v_proj') and old_attn.v_proj is not None:
                    v_weight = old_attn.v_proj.weight.data.clone()
                if hasattr(old_attn, 'dense') and old_attn.dense is not None:
                    dense_weight = old_attn.dense.weight.data.clone()
                    if hasattr(old_attn.dense, 'bias') and old_attn.dense.bias is not None:
                        dense_bias = old_attn.dense.bias.data.clone()
                
                # Create new attention
                new_attn = PhiFusedQKVAttention(model.config, layer_idx=idx)
                
                # Transfer captured weights
                if q_weight is not None and k_weight is not None and v_weight is not None:
                    fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    new_attn.qkv_proj.weight.data = fused_weight
                    transferred_count += 1
                    print(f"✓ Transferred QKV weights for layer {idx}")
                else:
                    print(f"⚠ Could not capture QKV weights for layer {idx}")
                
                if dense_weight is not None:
                    new_attn.dense.weight.data = dense_weight
                    if dense_bias is not None:
                        new_attn.dense.bias.data = dense_bias
                    print(f"✓ Transferred dense weights for layer {idx}")
                
                # Replace
                layer.self_attn = new_attn
                patched_count += 1
                print(f"✓ Patched layer {idx}: {old_type} -> PhiFusedQKVAttention")
        
        print(f"\n✓ Patched {patched_count} layers, transferred weights for {transferred_count} layers")
        return patched_count, transferred_count
        
    except Exception as e:
        print(f"⚠ Manual patching failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

def test_generation(model, tokenizer, prompt, max_new_tokens=30):
    """Test generation with proper error handling"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated, None
    except Exception as e:
        return None, str(e)

def main():
    print("🚀 Monkey Patch Experiment - Phi Model with Weight Transfer")
    print("="*50)
    
    # Apply patch
    if not apply_monkey_patch():
        print("⚠ Could not apply monkey patch")
        return
    
    model_name = "microsoft/phi-1_5"
    
    try:
        print(f"\n📥 Loading model: {model_name}")
        print("   This may take a few minutes...")
        
        # Load model with weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Patch and transfer weights
        print("\n🔧 Applying patch and transferring weights...")
        patched, transferred = patch_with_weight_transfer(model)
        
        print(f"\n✓ Model loaded and patched!")
        print(f"  Patched: {patched} layers")
        print(f"  Weights transferred: {transferred} layers")
        
        # Verify patch
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                first_attn = model.model.layers[0].self_attn
                attn_type = type(first_attn).__name__
                print(f"  First layer attention: {attn_type}")
                
                # Check if weights were transferred
                if hasattr(first_attn, 'qkv_proj'):
                    weight_sum = first_attn.qkv_proj.weight.data.sum().item()
                    print(f"  QKV weight sum (should be non-zero): {weight_sum:.4f}")
        except Exception as e:
            print(f"  ⚠ Could not verify: {e}")
        
        # Run inference
        print("\n" + "="*50)
        print("📝 Running Inference Test")
        print("="*50)
        
        prompts = [
            "The future of artificial intelligence is",
            "In the beginning,",
            "Machine learning is a"
        ]
        
        for prompt in prompts:
            print(f"\n📝 Prompt: {prompt}")
            
            generated, error = test_generation(model, tokenizer, prompt)
            
            if error:
                print(f"⚠ Generation failed: {error}")
            else:
                print(f"✨ Generated: {generated}")
        
        print("\n" + "="*50)
        print("✅ Experiment Completed!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            clear_patch_mapping()
            print("\n✓ Cleaned up monkey patch mapping")
        except:
            pass

if __name__ == "__main__":
    main()