from transformers import AutoModelForCausalLM, AutoConfig
import torch

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
# Load the model's configuration
print(f"Loading configuration for {model_name}...")
config = AutoConfig.from_pretrained(model_name)
print("\nModel Configuration:")
print(config)

# Load the model
print(f"\nLoading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("\nModel Structure:")
print(model)

# Specifically print the LM head if identifiable by a common name
if hasattr(model, 'lm_head'):
    print("\nLM Head Structure:")
    print(model.lm_head)
elif hasattr(model, 'output_embedding'): # some models might use a different name
    print("\nOutput Embedding/LM Head Structure:")
    print(model.output_embedding)
else:
    print("\nCould not automatically identify the LM head by common attribute names like 'lm_head' or 'output_embedding'. You may need to inspect the full model structure above.")

# Example of getting hidden_size directly from config, which we used in train.py
print(f"\nHidden size from config: {config.hidden_size}")
print(f"Vocabulary size from config: {config.vocab_size}")
