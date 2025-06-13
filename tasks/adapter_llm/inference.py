import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
from pathlib import Path # Added for Modal
import modal # Added for Modal

# Modal App Setup
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers", 
    "torch", 
    "huggingface-hub", 
    "accelerate" # Good to have for Hugging Face models
).add_local_python_source("models") # Crucial for CTM import

app = modal.App(name="adapter-ctm-inference-app") # Changed app name

MODEL_DIR = Path("/models") # Consistent with train.py
# Use the same volume name as in training to access checkpoints
volume = modal.Volume.from_name("adapter-ctm", create_if_missing=True) 

# Attempt to import ContinuousThoughtMachine
# This structure assumes 'models' directory is correctly added by add_local_python_source
try:
    from models.ctm import ContinuousThoughtMachine
except ImportError as e:
    print(f"Error importing ContinuousThoughtMachine: {e}")
    print("Ensure 'models' directory is in the Python path or add_local_python_source is working correctly.")
    # As a fallback for potential local non-Modal execution where paths might be tricky:
    if 'ContinuousThoughtMachine' not in globals():
        try:
            # Get the directory of the current file
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # Get the workspace root (assuming tasks/adapter_llm/inference.py structure)
            workspace_root = os.path.abspath(os.path.join(current_file_dir, '../..')) 
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
            from models.ctm import ContinuousThoughtMachine
            print("Successfully imported ContinuousThoughtMachine via sys.path modification.")
        except ImportError as e_fallback:
            raise ImportError(f"Failed to import ContinuousThoughtMachine via both direct and sys.path modification: {e_fallback}")


class CTMAdapter(nn.Module):
    def __init__(self, llm_hidden_dim, ctm_config):
        super().__init__()
        self.llm_hidden_dim = llm_hidden_dim
        
        # Option 1: Remove projections entirely - match train.py architecture
        self.ctm_core = ContinuousThoughtMachine(
            iterations=ctm_config['ctm_iterations'],
            d_model=ctm_config['ctm_d_model'],
            d_input=ctm_config['ctm_internal_attn_dim'], # CTM's internal attention data dim
            heads=ctm_config['ctm_heads'],
            n_synch_out=ctm_config['ctm_n_synch_out'],
            n_synch_action=ctm_config['ctm_n_synch_action'],
            synapse_depth=ctm_config['ctm_synapse_depth'],
            memory_length=ctm_config['ctm_memory_length'],
            deep_nlms=ctm_config['ctm_deep_nlms'],
            memory_hidden_dims=ctm_config['ctm_memory_hidden_dims'],
            do_layernorm_nlm=ctm_config['ctm_do_layernorm_nlm'],
            backbone_type=ctm_config['ctm_backbone_type'],
            positional_embedding_type=ctm_config['ctm_positional_embedding_type'],
            out_dims=llm_hidden_dim,
            prediction_reshaper=[-1, ctm_config['ctm_bottleneck_dim']],
            dropout=ctm_config['ctm_dropout'],
            dropout_nlm=ctm_config['ctm_dropout_nlm'],
            neuron_select_type=ctm_config['ctm_neuron_select_type'],
            n_random_pairing_self=ctm_config['ctm_n_random_pairing_self']
        )
        
        # Option 2: Keep a refinement layer - match train.py
        self.refinement = nn.Linear(llm_hidden_dim, llm_hidden_dim)
        
    def forward(self, x_llm):
        batch_size, seq_len, _ = x_llm.shape
        residual = x_llm
        
        # Send full hidden dims directly to CTM
        ctm_predictions, _, _ = self.ctm_core(x_llm)  # (B, L, 1536) → (B, 1536, 8)
        
        # Get final thought
        processed_features = ctm_predictions[:, :, -1]  # (B, llm_hidden_dim)
        
        # Broadcast to sequence
        processed_features = processed_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, llm_hidden_dim)
        
        # Option 1: Direct residual
        #return residual + processed_features
        
        # Option 2: Refined residual - match train.py
        delta_h = self.refinement(processed_features)
        return residual + delta_h

def run_ctm_inference(
    base_model_name: str,
    adapter_checkpoint_path: str,
    ctm_config: dict,
    prompt_text: str,
    tokenizer_name: str = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Loads a base LLM, a CTM adapter, and performs inference.

    Args:
        base_model_name (str): Name of the Hugging Face base model.
        adapter_checkpoint_path (str): Path to the trained CTM adapter checkpoint (.pth).
        ctm_config (dict): Configuration dictionary for the CTMAdapter.
        prompt_text (str): The input prompt for generation.
        tokenizer_name (str, optional): Name of the tokenizer if different from base_model_name. Defaults to None.
        max_new_tokens (int, optional): Maximum new tokens to generate. Defaults to 100.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling top_p. Defaults to 0.9.

    Returns:
        str: The generated text (excluding the prompt).
    """
    if tokenizer_name is None:
        tokenizer_name = base_model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token_id})")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is not None and base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set base_model.config.pad_token_id to {tokenizer.pad_token_id}")
    
    llm_hidden_dim = base_model.config.hidden_size

    # Instantiate CTMAdapter
    adapter = CTMAdapter(llm_hidden_dim=llm_hidden_dim, ctm_config=ctm_config)
    
    # Load adapter weights
    if os.path.exists(adapter_checkpoint_path):
        adapter.load_state_dict(torch.load(adapter_checkpoint_path, map_location=device))
        print(f"Loaded CTM adapter weights from {adapter_checkpoint_path}")
    else:
        print(f"Warning: Adapter checkpoint path not found: {adapter_checkpoint_path}. Using initialized adapter.")

    base_model.to(device)
    adapter.to(device)
    base_model.eval()
    adapter.eval()

    # Prepare inputs
    prompt_ids = tokenizer.encode(prompt_text + tokenizer.eos_token, return_tensors="pt", truncation=True).to(device)
    
    generated_ids = prompt_ids.clone()

    print(f"Starting generation with prompt (length {prompt_ids.shape[1]}): \"{prompt_text}\"")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = base_model(input_ids=generated_ids, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            
            adapted_hidden_states = adapter(last_hidden_states)
            
            next_token_logits = base_model.lm_head(adapted_hidden_states)[:, -1, :]

            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)

            if 0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                
                if torch.all(probs == 0):
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                else:
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                if temperature == 0:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                print("EOS token generated. Stopping.")
                break
        
    generated_sequence = generated_ids[0]
    # Decode only the newly generated tokens
    generated_text_only = tokenizer.decode(generated_sequence[prompt_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text_only

def run_raw_inference(
    base_model_name: str,
    prompt_text: str,
    tokenizer_name: str = None,
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Loads a base LLM and performs inference without any adapter.

    Args:
        base_model_name (str): Name of the Hugging Face base model.
        prompt_text (str): The input prompt for generation.
        tokenizer_name (str, optional): Name of the tokenizer if different from base_model_name. Defaults to None.
        max_new_tokens (int, optional): Maximum new tokens to generate. Defaults to 100.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling top_p. Defaults to 0.9.

    Returns:
        str: The generated text (excluding the prompt).
    """
    if tokenizer_name is None:
        tokenizer_name = base_model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token_id})")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is not None and base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set base_model.config.pad_token_id to {tokenizer.pad_token_id}")

    base_model.to(device)
    base_model.eval()

    # Prepare inputs
    prompt_ids = tokenizer.encode(prompt_text + tokenizer.eos_token, return_tensors="pt", truncation=True).to(device)
    
    generated_ids = prompt_ids.clone()

    print(f"Starting RAW generation with prompt (length {prompt_ids.shape[1]}): \"{prompt_text}\"")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = base_model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)

            if 0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                
                if torch.all(probs == 0):
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                else:
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                if temperature == 0:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                print("EOS token generated. Stopping.")
                break
        
    generated_sequence = generated_ids[0]
    # Decode only the newly generated tokens
    generated_text_only = tokenizer.decode(generated_sequence[prompt_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text_only

# CTM configuration matching exactly with train.py
DEFAULT_CTM_CONFIG_FOR_INFERENCE = {
        "ctm_bottleneck_dim": 896,          # Would become 1536
        "ctm_d_model": 896,                # Should be >= bottleneck_dim, so maybe 1536 or 2048
        "ctm_internal_attn_dim": 896,       # Could become 1536 for consistency
        "ctm_iterations": 32,                # More thinking steps
        "ctm_heads": 8,                     # Enable attention
        "ctm_n_synch_out": 512,
        "ctm_n_synch_action": 512,
        "ctm_synapse_depth": 2,             # Deeper synapses
        "ctm_memory_length": 8,             # Longer memory
        "ctm_deep_nlms": True,
        "ctm_memory_hidden_dims": 32,
        "ctm_do_layernorm_nlm": False,
        "ctm_dropout": 0.1,
        "ctm_dropout_nlm": 0.1,
        "ctm_neuron_select_type": 'random-pairing',
        "ctm_n_random_pairing_self": 4,
        "ctm_backbone_type": "token-processing",  # Use new token-processing backbone
        "ctm_positional_embedding_type": "sequence-rotational",
}


@app.function(gpu="A10G", image=image, volumes={str(MODEL_DIR): volume}, timeout=1800)
def modal_run_inference_entrypoint(
    prompt_text: str,
    run_id: str, 
    epoch_number: int,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    checkpoint_filename: str = "adapter_checkpoint.pth",
    ctm_config_dict: dict = None, # Will default to DEFAULT_CTM_CONFIG_FOR_INFERENCE if None
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    print(f"Modal Inference Started for run_id: {run_id}, epoch: {epoch_number}")
    print(f"Attempting to load ContinuousThoughtMachine: {ContinuousThoughtMachine}")

    actual_ctm_config = ctm_config_dict if ctm_config_dict is not None else DEFAULT_CTM_CONFIG_FOR_INFERENCE

    # Construct checkpoint path within the Modal volume
    # Consistent with how checkpoints are saved in train.py: MODEL_DIR / "checkpoints" / run_id / f"epoch_{epoch_number}" / checkpoint_filename
    checkpoint_path_on_volume = MODEL_DIR / "checkpoints" / run_id / f"epoch_{epoch_number}" / checkpoint_filename
    
    print(f"Constructed checkpoint path: {checkpoint_path_on_volume}")

    if not os.path.exists(checkpoint_path_on_volume):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path_on_volume} on the Modal volume.")
        # List contents of the expected directory for debugging
        expected_dir = MODEL_DIR / "checkpoints" / run_id / f"epoch_{epoch_number}"
        if os.path.exists(expected_dir):
            print(f"Contents of {expected_dir}: {os.listdir(expected_dir)}")
        else:
            print(f"Expected directory {expected_dir} does not exist.")
        
        run_id_dir = MODEL_DIR / "checkpoints" / run_id
        if os.path.exists(run_id_dir):
            print(f"Contents of {run_id_dir}: {os.listdir(run_id_dir)}")
        else:
            print(f"Run ID directory {run_id_dir} does not exist.")

        checkpoints_dir = MODEL_DIR / "checkpoints"
        if os.path.exists(checkpoints_dir):
            print(f"Contents of {checkpoints_dir}: {os.listdir(checkpoints_dir)}")
        else:
            print(f"Base checkpoints directory {checkpoints_dir} does not exist.")
        # Provide a more informative error or allow proceeding with uninitialized adapter for certain tests
        # For now, we'll let run_ctm_inference handle the warning if path doesn't exist, 
        # but this check is good for Modal context.
        # return # Or raise error

    generated_text = run_ctm_inference(
        base_model_name=base_model_name,
        adapter_checkpoint_path=str(checkpoint_path_on_volume), 
        ctm_config=actual_ctm_config, 
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    full_output = prompt_text + generated_text
    return prompt_text, generated_text, full_output

@app.function(gpu="A10G", image=image, volumes={str(MODEL_DIR): volume}, timeout=1800)
def modal_run_raw_inference_entrypoint(
    prompt_text: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Modal function to run raw inference without adapter.
    """
    print(f"Modal Raw Inference Started for base model: {base_model_name}")
    
    generated_text = run_raw_inference(
        base_model_name=base_model_name,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    full_output = prompt_text + generated_text
    return prompt_text, generated_text, full_output

@app.local_entrypoint()
def main():
    # This main function is for triggering the Modal job from your local machine.
    # You need to provide a valid run_id and epoch_number for a checkpoint that exists on the Modal Volume.
    
    example_prompt = """
Below is an instruction to modify a code file along with the code file. Apply these instructions and return the adapted code:

Instruction:
Transform the initial error message script into a YAML configuration schema. Define a comprehensive schema using the 'yaml' library that includes sections for MQTT settings, GPIO modules, digital inputs, and digital outputs, each with specific attributes and default values.

Code:
pi_mqtt_gpio/__init__.py
```Python
import sys
print("FATAL ERROR: The file at pi_mqtt_gpio/__init__.py should be replaced us"
"ing 'make schema' before packaging.")
sys.exit(1)



Now return the full adjusted code, and just the code without any explanation.
"""
    
    # Replace with actual run_id and epoch_number from a completed training run
    # For example, if your train.py saved a checkpoint for:
    # run_id = "run_Qwen_Adapter_0.5B_Instruct__CTM__20231027-120000" (example format)
    # epoch_number = 1 (training script saves epoch_1, epoch_2 etc.)
    
    # --- Parameters for testing --- 
    # You MUST replace these with valid values from your training runs
    test_run_id = "run_Qwen_Adapter_0.5B_Instruct__CTM__1749351122" # Placeholder - replace with your actual run_id
    test_epoch_number = 1 # Placeholder - replace with your actual epoch number

    print("="*60)
    print("COMPARISON: RAW MODEL vs ADAPTER MODEL")
    print("="*60)

    # First, run raw inference for baseline
    print("\n1. Running RAW model inference (no adapter)...")
    try:
        raw_prompt_text, raw_generated_text, raw_full_output = modal_run_raw_inference_entrypoint.remote(
            prompt_text=example_prompt
        )
        print(f"RAW Model - Prompt: {raw_prompt_text}")
        print(f"RAW Model - Generated: {raw_generated_text}")
        print("-" * 40)
    except Exception as e:
        print(f"Error in raw inference: {e}")
        raw_generated_text = "ERROR"

    # Then, run adapter inference
    print(f"\n2. Running ADAPTER model inference for run_id: {test_run_id}, epoch: {test_epoch_number}")
    print("NOTE: This requires a checkpoint to exist on the Modal volume at the specified path.")
    print(f"Expected path: /models/checkpoints/{test_run_id}/epoch_{test_epoch_number}/adapter_checkpoint.pth")
    
    try:
        adapter_prompt_text, adapter_generated_text, adapter_full_output = modal_run_inference_entrypoint.remote(
            prompt_text=example_prompt,
            run_id=test_run_id, 
            epoch_number=test_epoch_number,
            max_new_tokens=1000,
            # ctm_config_dict=DEFAULT_CTM_CONFIG_FOR_INFERENCE # Optional, if defaults are fine
        )
        print(f"ADAPTER Model - Prompt: {adapter_prompt_text}")
        print(f"ADAPTER Model - Generated: {adapter_generated_text}")
        print("-" * 40)
    except Exception as e:
        print(f"Error in adapter inference: {e}")
        print("Please ensure the Modal daemon is running, you are logged in, and the specified checkpoint exists on the volume.")
        adapter_generated_text = "ERROR"

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Prompt: {example_prompt}")
    print(f"\nRAW Model Output:\n{raw_generated_text}")
    print(f"\nADAPTER Model Output:\n{adapter_generated_text}")
    print("="*60)
