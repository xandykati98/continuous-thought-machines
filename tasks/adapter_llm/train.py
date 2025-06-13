# Load model directly
from pathlib import Path
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("transformers", "datasets", "torch", "huggingface-hub", "muon-optimizer", "wandb").add_local_python_source("models")

app = modal.App(name="adapter-ctm-training-app", image=image)

MODEL_DIR = Path("/models")

volume = modal.Volume.from_name("adapter-ctm", create_if_missing=True)
@app.function(gpu="a100", image=image, timeout=3600*23, volumes={MODEL_DIR: volume})
def modal__train_adapter_ctm():
    from models.ctm import ContinuousThoughtMachine
    print(ContinuousThoughtMachine)
    import time
    import json # Added for saving sample output
    import os # Added for creating directories
    import random # Added for random sampling
    from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from huggingface_hub import login
    import wandb # Added for W&B integration
    # Import CTM
    # from models.ctm import ContinuousThoughtMachine # Adjusted path
    # Assuming models/ is in python path for Modal, or use relative if appropriate for local execution context.
    # For Modal, if workspace is mounted, 'from models.ctm import ContinuousThoughtMachine' might work if run from root.
    # Using placeholder, as actual import path depends on Modal execution context and file structure.
    # This was added by the AI: from ..models.ctm import ContinuousThoughtMachine # Added for CTM



    
    wandb.login(key="5c0d2d6b1fcad21af4e0cc3894c119285c4ddae5")
    try:
        login(token="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") 
        print("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        print(f"Hugging Face Hub login failed: {e}. Dataset loading might fail if it's private.")

    # Load dataset
    try:
        ds = load_dataset("PrimeIntellect/real-world-swe-problems", split="train") # Using train split
        print(f"Dataset loaded. Number of examples: {len(ds)}")

        # For testing, use a much smaller subset of the dataset
        max_dataset_size_for_testing = 50000000000 # Adjust this value as needed for testing
        if len(ds) > max_dataset_size_for_testing:
            ds = ds.select(range(max_dataset_size_for_testing))
            print(f"Reduced dataset to first {max_dataset_size_for_testing} examples for testing. New size: {len(ds)}")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Exiting due to dataset loading failure.")
        exit()


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    # Add padding token if it doesn't exist. For Causal LMs, often pad_token = eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.eos_token_id # This line caused an error if model not defined yet
        print("Set tokenizer.pad_token to tokenizer.eos_token")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")

    # Initialize W&B
    # Generate a unique run ID for this training session
    # Update run_id for CTM
    run_id = f"run_Qwen_Adapter_0.5B_Instruct__CTM__updownzero_with_norm_low_certainties_adamw__{int(time.time())}" 
    wandb.init(project="adapter_llm_training", name=run_id, config={
        "learning_rate": 1e-4,
        "epochs": 3, 
        "batch_size": 3, 
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        # "adapter_bottleneck": 1024, # This is for the old linear adapter, new one is ctm_bottleneck_dim
        "dataset": "PrimeIntellect/real-world-swe-problems",
        "run_id": run_id,
        "wandb_api_key": "5c0d2d6b1fcad21af4e0cc3894c119285c4ddae5",
        # CTM Adapter Hyperparameters
        "adapter_type": "CTM",
        "ctm_bottleneck_dim": 896,          # Would become 1536
        "ctm_d_model": 896*2,                # Should be >= bottleneck_dim, so maybe 1536 or 2048
        "ctm_internal_attn_dim": 896,       # Could become 1536 for consistency
        "ctm_iterations": 80,                # More thinking steps
        "ctm_heads": 8,                     # Enable attention
        "ctm_n_synch_out": 512,
        "ctm_n_synch_action": 512,
        "ctm_synapse_depth": 8,             # Deeper synapses
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
        # Certainty weighting hyperparameters
        "use_certainty_weighting": True,
        "certainty_weighting_mode": "final",  # Options: 'final', 'max', 'avg'
        "certainty_scaling_factor": 0.5,    # Scale the certainty values
        "optimizer": "adamw",  # Track which optimizer you're using
    })

    class CTMAdapter(nn.Module):
        def __init__(self, llm_hidden_dim, ctm_config):
            super().__init__()
            self.llm_hidden_dim = llm_hidden_dim
            
            # CTM Core - the main thinking machine
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
            
            # Projection layers with normalization
            self.down_proj = nn.Sequential(
                nn.Linear(llm_hidden_dim, llm_hidden_dim),
                nn.LayerNorm(llm_hidden_dim)
            )
            self.up_proj = nn.Sequential(
                nn.LayerNorm(llm_hidden_dim),
                nn.Linear(llm_hidden_dim, llm_hidden_dim)
            )
            
            # Custom initialization for projections only (CTM has its own init)
            self._init_adapter_weights()
            
            # Add hyperparameters for certainty weighting
            self.use_certainty_weighting = ctm_config.get('use_certainty_weighting', True)
            self.certainty_weighting_mode = ctm_config.get('certainty_weighting_mode', 'final')  # 'final', 'max', 'avg'
            self.certainty_scaling_factor = ctm_config.get('certainty_scaling_factor', 1.0)
            
        def _init_adapter_weights(self):
            """Initialize adapter weights for stable training."""
            # Zero init the linear layers
            for module in [self.down_proj, self.up_proj]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
            print("Initialized CTMAdapter projections with zero weights")
            
        def forward(self, x_llm):
            batch_size, seq_len, _ = x_llm.shape
            residual = x_llm
            
            # Apply down_proj before feeding to CTM core
            down_projected = self.down_proj(x_llm)
            
            # Send down-projected features to CTM - NOW capture certainties!
            ctm_predictions, ctm_certainties, _ = self.ctm_core(down_projected)  # (B, L, T), (B, 2, T), _
            
            # Get final thought
            processed_features = ctm_predictions[:, :, -1]  # (B, L)
            
            # Broadcast to sequence length
            processed_features = processed_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, H)
            
            # Apply up_proj to CTM output
            delta_h = self.up_proj(processed_features)
            
            # Use certainty to weight the delta
            if self.use_certainty_weighting:
                # Extract certainty values (certainties[:, 1, :] is the actual certainty, not entropy)
                certainty_values = ctm_certainties[:, 1, :]  # (B, T)
                
                if self.certainty_weighting_mode == 'final':
                    # Use certainty from final iteration
                    certainty_weight = certainty_values[:, -1]  # (B,)
                elif self.certainty_weighting_mode == 'max':
                    # Use maximum certainty across all iterations
                    certainty_weight = certainty_values.max(dim=1)[0]  # (B,)
                elif self.certainty_weighting_mode == 'avg':
                    # Use average certainty across all iterations
                    certainty_weight = certainty_values.mean(dim=1)  # (B,)
                else:
                    raise ValueError(f"Unknown certainty_weighting_mode: {self.certainty_weighting_mode}")
                
                # Scale and reshape certainty weight to broadcast with delta_h
                certainty_weight = certainty_weight * self.certainty_scaling_factor  # (B,)
                certainty_weight = certainty_weight.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
                
                # Apply certainty weighting to delta
                delta_h = delta_h * certainty_weight
            
            return residual + delta_h

    # Hidden size from user comment/model config
    hidden_size = model.config.hidden_size # Get hidden size from the loaded model's config
    print(f"Using hidden_size: {hidden_size}")

    
    # Instantiate the CTM adapter
    ctm_adapter_config = {k: v for k, v in wandb.config.items() if k.startswith('ctm_')}
    adapter = CTMAdapter(llm_hidden_dim=hidden_size, ctm_config=ctm_adapter_config)


    # Training Loop Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    adapter.to(device)

    print("\nModel Structure:")
    print(model)
    print("Adapter Structure:")
    print(adapter)
    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False
    print("Froze base model parameters.")

    # Adapter parameters are trainable by default as it's a new module.
    # Ensure adapter parameters require grad (should be true by default for new nn.Module)
    for param in adapter.parameters():
        param.requires_grad = True
    print("Adapter parameters are trainable.")

    print(f"Tokenizer model max length: {tokenizer.model_max_length}")

    preprocess_sizes = []
    # Add this function before the preprocess_function definition
    def find_global_max_length(dataset, tokenizer):
        """Find the maximum sequence length across the entire dataset"""
        print("Computing global maximum sequence length...")
        max_length = 0
        
        for i, example in enumerate(dataset):
            if i % 2500 == 0:  # Progress indicator
                print(f"Processed {i}/{len(dataset)} examples...")
                
            prompt_with_eos = example["prompt"] + tokenizer.eos_token 
            full_text = prompt_with_eos + example["gold_standard_solution"] + tokenizer.eos_token
            
            # Tokenize without truncation to get true length
            tokenized = tokenizer(full_text, truncation=False, padding=False)
            current_length = len(tokenized["input_ids"])
            max_length = max(max_length, current_length)
        
        print(f"Global maximum sequence length: {max_length} tokens")
        return max_length

    # Updated preprocess_function to use global max length
    def preprocess_function(examples):
        processed_examples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for prompt, solution in zip(examples["prompt"], examples["gold_standard_solution"]):
            prompt_with_eos = prompt + tokenizer.eos_token 
            full_text = prompt_with_eos + solution + tokenizer.eos_token

            # Tokenize the full text using global max length
            tokenized_full = tokenizer(
                full_text, 
                max_length=global_max_length,  # Use the computed global max length
                padding="max_length",
                truncation=True, 
                return_attention_mask=True
            )

            # Tokenize the prompt part separately to find its length
            tokenized_prompt = tokenizer(
                prompt_with_eos,
                truncation=True,
                padding=False,
                add_special_tokens=False # Important: eos_token already added
            )
            
            prompt_tokens_length = len(tokenized_prompt["input_ids"])

            # Create labels: initially a copy of input_ids
            labels = list(tokenized_full["input_ids"]) # Make it a list for modification

            # Mask out the prompt tokens in the labels
            # The model should only learn to predict the solution part
            for i in range(prompt_tokens_length):
                if i < len(labels): # Ensure we don't go out of bounds if full_text was truncated shorter than prompt
                    labels[i] = -100

            processed_examples["input_ids"].append(tokenized_full["input_ids"])
            processed_examples["attention_mask"].append(tokenized_full["attention_mask"])
            processed_examples["labels"].append(labels) # Use the modified labels

        return processed_examples

    # Split dataset and prepare fixed sample for generation
    num_total_examples = len(ds)
    num_test_examples = 1000
    fixed_test_sample_prompt_ids = None
    fixed_test_sample_gold_solution = "N/A"
    fixed_test_sample_raw_prompt = "N/A"
    test_ds = None

    if num_total_examples <= num_test_examples:
        print(f"Warning: Total dataset size ({num_total_examples}) is less than or equal to requested test set size ({num_test_examples}).")
        if num_total_examples > 1:
            num_test_examples = min(num_test_examples, num_total_examples // 2)
            train_ds = ds.select(range(num_test_examples, num_total_examples))
            test_ds = ds.select(range(num_test_examples))
        else:
            train_ds = ds
            test_ds = None
    else:
        test_ds = ds.select(range(num_test_examples))
        train_ds = ds.select(range(num_test_examples, num_total_examples))

    test_sample_index = 2
    train_sample_index = 2
    if test_ds and len(test_ds) > 0:
        print(f"Test set size: {len(test_ds)}")
        # Prepare the fixed sample for generation
        raw_sample = test_ds[test_sample_index] # Take the first sample from the test set
        fixed_test_sample_raw_prompt = raw_sample["prompt"]
        fixed_test_sample_gold_solution = raw_sample["gold_standard_solution"]
        # Tokenize only the prompt part for generation input
        fixed_test_sample_prompt_ids = tokenizer(fixed_test_sample_raw_prompt + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=32768).input_ids.to(device)
        print(f"Prepared a fixed sample from test_ds for qualitative checking. Prompt length: {fixed_test_sample_prompt_ids.shape[1]} tokens.")
    elif train_ds and len(train_ds) > 0: # Fallback to train_ds if test_ds is empty but train_ds is not
        print("Warning: Test set is empty or too small. Using a sample from the training set for qualitative checking.")
        raw_sample = train_ds[train_sample_index]
        fixed_test_sample_raw_prompt = raw_sample["prompt"]
        fixed_test_sample_gold_solution = raw_sample["gold_standard_solution"]
        fixed_test_sample_prompt_ids = tokenizer(fixed_test_sample_raw_prompt + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=32768).input_ids.to(device)
        print(f"Prepared a fixed sample from train_ds for qualitative checking. Prompt length: {fixed_test_sample_prompt_ids.shape[1]} tokens.")
    else:
        print("No test or train samples available to create a fixed generation sample.")

    print(f"Training set size: {len(train_ds) if train_ds else 0}")

    # Compute global max length BEFORE preprocessing
    if train_ds and len(train_ds) > 0:
        global_max_length = find_global_max_length(train_ds, tokenizer)
        
        # Cap it to model's context window as safety measure
        model_max_length = 32768  # Your model's context window
        if global_max_length > model_max_length:
            print(f"Warning: Dataset max length ({global_max_length}) exceeds model context window ({model_max_length}). Capping to model limit.")
            global_max_length = model_max_length
        
        print(f"Using global max length: {global_max_length} tokens")
    else:
        global_max_length = 4096  # Fallback

    print("Starting dataset preprocessing for training data...")
    if train_ds and len(train_ds) > 0:
        tokenized_train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
        print("Training dataset preprocessing finished.")
        print(f"Preprocess sizes: {preprocess_sizes}")
    else:
        print("Skipping training dataset preprocessing as train_ds is empty or None.")
        tokenized_train_ds = None

    # Function to generate and log a sample
    def generate_and_log_sample(model, adapter, tokenizer, device, epoch, run_id, epoch_completion, current_loss, test_ds, train_ds, max_new_tokens=30000, base_filename="generation_sample.json"):
        # global test_ds, train_ds # Ensure we're using the globally defined datasets
        print('Generating and logging sample INSIDE...')
        # Determine which dataset to use for sampling
        sample_source_ds = None
        if test_ds and len(test_ds) > 0:
            sample_source_ds = test_ds
            source_name = "test_ds"
        elif train_ds and len(train_ds) > 0:
            sample_source_ds = train_ds
            source_name = "train_ds"
        else:
            print("Skipping sample generation as no dataset is available.")
            return

        print('Selecting a random sample...')
        # Select a random sample
        sample_index = random.randint(0, len(sample_source_ds) - 1)
        print('Sample selected.')
        raw_sample = sample_source_ds[sample_index]
        raw_prompt_text = raw_sample["prompt"]
        gold_solution = raw_sample["gold_standard_solution"]
        
        print('Tokenizing prompt...')
        # Tokenize only the prompt part for generation input
        prompt_ids = tokenizer(raw_prompt_text + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=32768).input_ids.to(device)
        print('Prompt tokenized.')

        if prompt_ids is None: # Should not happen if a sample was selected
            print("Skipping sample generation as no prompt is available after selection.")
            return

        model.eval()
        adapter.eval()
        generated_ids = prompt_ids.clone() # Start with the prompt
        print('Starting generation...')
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if _ % 150 == 0:
                    print('Generating next token...', _)
                # Get model outputs (hidden states)
                outputs = model(input_ids=generated_ids, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                
                # Apply adapter
                adapted_hidden_states = adapter(last_hidden_states)
                
                # Get logits from lm_head
                next_token_logits = model.lm_head(adapted_hidden_states)[:, -1, :]
                
                # --- Start of sampling modification ---
                temperature = 0.7  # Example: Make this configurable
                top_p = 0.9        # Example: Make this configurable

                if temperature > 0:
                    scaled_logits = next_token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                else: # temperature == 0 is equivalent to greedy on original logits
                    probs = torch.softmax(next_token_logits, dim=-1)


                if 0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Create a mask for tokens to remove
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the mask: always keep at least the most probable token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask in the original unsorted indices
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    
                    # Renormalize if necessary, though multinomial handles unnormalized distributions if sum > 0
                    if torch.all(probs == 0): # Fallback if all probabilities became zero
                        # Fallback to argmax on original logits if filtering removed all tokens
                        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    else:
                        probs = probs / probs.sum(dim=-1, keepdim=True) # Ensure probs sum to 1 for multinomial
                        next_token_id = torch.multinomial(probs, num_samples=1)
                else: # If top_p is not used (e.g. 0 or 1.0), just use temperature scaled probs or fall back to argmax for temp 0
                    if temperature == 0: # Equivalent to greedy
                         next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    else:
                         next_token_id = torch.multinomial(probs, num_samples=1)
                # --- End of sampling modification ---
                
                # Append token to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

                # Stop if EOS token is generated
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        generated_text = tokenizer.decode(generated_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        full_generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print("\n--- Sample Generation ---")
        print(f"\033[93mPrompt: {raw_prompt_text}\033[0m")
        print(f"\033[92mGold Solution: {gold_solution}\033[0m")
        print(f"\033[95mModel Generated: {generated_text}\033[0m")
        print("--- End Sample Generation ---\n")

        sample_output = {
            "epoch": epoch,
            "run_id": run_id,
            "epoch_completion": f"{epoch_completion:.2%}",
            "current_loss": f"{current_loss:.4f}" if current_loss is not None else "N/A",
            "prompt": raw_prompt_text,
            "gold_standard_solution": gold_solution,
            "model_generated_solution": generated_text,
            "full_generated_sequence_incl_prompt": full_generated_text,
            "sample_source": source_name,
            "sample_index": sample_index
        }
        
        # Define base directory for this run's outputs on the volume
        run_output_dir_on_volume = MODEL_DIR / run_id
        
        # Path for the main sample file (e.g., generation_sample.json, final_generation_sample.json)
        main_sample_filepath_on_volume = run_output_dir_on_volume / base_filename # base_filename is passed as arg
        
        # Path for detailed logs directory
        detailed_logs_dir_on_volume = run_output_dir_on_volume / "detailed_generation_logs"

        # Create directories on the volume if they don't exist
        try:
            os.makedirs(detailed_logs_dir_on_volume, exist_ok=True) # Ensures run_output_dir_on_volume is also created
            print(f"Ensured directory exists for sample logs: {detailed_logs_dir_on_volume}")
        except Exception as e:
            print(f"Error creating directory on volume {detailed_logs_dir_on_volume}: {e}")
            # Depending on desired behavior, could raise e or return

        # Save the main sample generation to the volume
        try:
            with open(main_sample_filepath_on_volume, 'w', encoding='utf-8') as f:
                json.dump(sample_output, f, indent=4, ensure_ascii=False)
            print(f"Saved latest sample generation to {main_sample_filepath_on_volume} on Modal Volume.")
        except Exception as e:
            print(f"Error saving sample generation to {main_sample_filepath_on_volume} on Modal Volume: {e}")

        # Construct filename for the detailed log on the volume
        # Using a consistent naming convention for detailed logs
        time_str = time.strftime('%Y%m%d-%H%M%S')
        detailed_log_actual_filename = f"log_epoch{epoch}_run{run_id}_time{time_str}.json"
        detailed_log_filepath_on_volume = detailed_logs_dir_on_volume / detailed_log_actual_filename
        
        # Save detailed record to the volume
        try:
            with open(detailed_log_filepath_on_volume, 'w', encoding='utf-8') as f:
                json.dump(sample_output, f, indent=4, ensure_ascii=False)
            print(f"Saved detailed sample generation to {detailed_log_filepath_on_volume} on Modal Volume.")
        except Exception as e:
            print(f"Error saving detailed sample generation to {detailed_log_filepath_on_volume} on Modal Volume: {e}")

        # Commit changes to the volume after all file operations for this sample
        try:
            volume.commit()
            print(f"Successfully committed sample logs for run {run_id}, epoch {epoch} to Modal Volume.")
        except Exception as e:
            print(f"Error committing sample logs to Modal Volume: {e}")
        
        # Log to W&B
        if wandb.run:
            wandb.log({"generation_sample": sample_output, "current_loss_at_sample": current_loss if current_loss is not None else -1})


        # Set back to train mode if they were in train mode before
        # This is handled by the main loop which calls adapter.train() each epoch


    # Data Collator for Causal Language Modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    batch_size = wandb.config.batch_size
    # Create DataLoader
    # Adjust batch_size as per your GPU memory
    if tokenized_train_ds and len(tokenized_train_ds) > 0:
        train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        print(f"Train DataLoader created with batch size {batch_size}. Number of batches: {len(train_dataloader)}")
    else:
        print("Train DataLoader not created as there is no tokenized training data.")
        # Potentially exit or handle this scenario if training cannot proceed
        exit("Exiting: No data for training DataLoader.")


    # Optimizer - Using AdamW 
    optimizer = optim.AdamW(adapter.parameters(), lr=wandb.config.learning_rate)

    # Training configuration
    num_epochs = wandb.config.epochs

    # Prepare DataLoader for training
    if tokenized_train_ds:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # This is redundant, data_collator is already defined
        train_dataloader = DataLoader(tokenized_train_ds, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=data_collator)
        print(f"Created DataLoader with batch size {wandb.config.batch_size}. Number of batches: {len(train_dataloader)}")
    else:
        print("No training data to load into DataLoader. Exiting.")
        exit()


    all_losses = []
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        adapter.train() 
        model.lm_head.eval() # Keep lm_head in eval as it's part of the frozen model section

        # Generate sample at the start of each epoch
        print(f"Generating sample for epoch {epoch+1} start...")
        # Pass None for current_loss as it's not available yet
        generate_and_log_sample(model, adapter, tokenizer, device, epoch=epoch, run_id=run_id, epoch_completion=0.0, current_loss=None, test_ds=test_ds, train_ds=train_ds)


        epoch_losses = []
        times = []
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            time_start = time.time()

            # batch is a dict of tensors from DataCollatorForLanguageModeling
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # Labels are created by the collator

            # Forward pass through the base model to get hidden states
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            last_hidden_states = outputs.hidden_states[-1]
            
            adapted_hidden_states = adapter(last_hidden_states)
            
            logits = model.lm_head(adapted_hidden_states)
            
            # Shift logits and labels for next token prediction
            # Labels provided by DataCollatorForLanguageModeling (mlm=False) are already suitable for Causal LM,
            # meaning they are typically a shifted version of input_ids or a direct copy where padded tokens are -100.
            # The standard way is to predict the next token, so labels should align with logits after shifting logits.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss, DataCollatorForLanguageModeling pads labels with -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            optimizer.step()
            
            # Log loss to W&B
            if wandb.run:
                wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch_idx": batch_idx})


            epoch_losses.append(loss.item())
            time_end = time.time()
            times.append(time_end - time_start)
            
            avg_time = sum(times) / len(times) if times else 0
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Batch Time: {(time_end - time_start):.2f}s, Avg Batch Time: {avg_time:.2f}s")

            # Generate sample every 50 batches (or as configured)
            if (batch_idx + 1) % 10000 == 0: # Generate sample periodically
                print(f"Generating sample for epoch {epoch+1}, batch {batch_idx+1}...")
                epoch_completion = (batch_idx + 1) / len(train_dataloader)
                generate_and_log_sample(model, adapter, tokenizer, device, epoch=epoch, run_id=run_id, epoch_completion=epoch_completion, current_loss=loss.item(), test_ds=test_ds, train_ds=train_ds, max_new_tokens=2000)

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        all_losses.extend(epoch_losses) # Store all batch losses
        print(f"Epoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")
        if wandb.run:
            wandb.log({"average_epoch_loss": avg_epoch_loss, "epoch": epoch})

        # Save checkpoint to Modal Volume at the end of each epoch
        checkpoint_base_dir_on_volume = MODEL_DIR / "checkpoints"
        run_checkpoint_dir_on_volume = checkpoint_base_dir_on_volume / run_id
        # Use epoch+1 for 1-indexed epoch folder name, matching print statements
        epoch_checkpoint_dir_on_volume = run_checkpoint_dir_on_volume / f"epoch_{epoch+1}"
        
        try:
            os.makedirs(epoch_checkpoint_dir_on_volume, exist_ok=True)
            print(f"Ensured checkpoint directory exists: {epoch_checkpoint_dir_on_volume}")
        except Exception as e:
            print(f"Error creating checkpoint directory {epoch_checkpoint_dir_on_volume} on volume: {e}")
            # Continue attempt to save, might fail if dir creation failed but path was already there

        checkpoint_filename = "adapter_checkpoint.pth"
        checkpoint_path_on_volume = epoch_checkpoint_dir_on_volume / checkpoint_filename

        try:
            torch.save(adapter.state_dict(), checkpoint_path_on_volume)
            print(f"Saved adapter checkpoint for epoch {epoch+1} to {checkpoint_path_on_volume} on Modal Volume.")
            volume.commit() # Persist the checkpoint
            print(f"Successfully committed checkpoint for epoch {epoch+1} to Modal Volume.")
            if wandb.run:
                # Log the string representation of the Path object
                wandb.log({"epoch_checkpoint_path": str(checkpoint_path_on_volume), "epoch": epoch})
        except Exception as e:
            print(f"Error saving or committing checkpoint for epoch {epoch+1} to {checkpoint_path_on_volume}: {e}")


    print("Training loop finished.")
    print(f"All recorded batch losses (first 100 if many): {all_losses[:100]}")
    print("You can now use 'all_losses' list to plot the training curve.")

    # if fixed_test_sample_prompt_ids is not None:
    print("Generating final sample after training completion...")
    # Use the last available loss or average epoch loss for the final sample
    final_loss = epoch_losses[-1] if epoch_losses else (all_losses[-1] if all_losses else None)
    generate_and_log_sample(model, adapter, tokenizer, device, epoch=num_epochs -1, run_id=run_id, epoch_completion=1.0, current_loss=final_loss, test_ds=test_ds, train_ds=train_ds, base_filename="final_generation_sample.json")


    if wandb.run:
        wandb.finish()


# Model Structure:
# Qwen2ForCausalLM(
#   (model): Qwen2Model(
#     (embed_tokens): Embedding(151936, 896)
#     (layers): ModuleList(
#       (0-23): 24 x Qwen2DecoderLayer(
#         (self_attn): Qwen2Attention(
#           (q_proj): Linear(in_features=896, out_features=896, bias=True)
#           (k_proj): Linear(in_features=896, out_features=128, bias=True)
#           (v_proj): Linear(in_features=896, out_features=128, bias=True)
#           (o_proj): Linear(in_features=896, out_features=896, bias=False)
#         )
#         (mlp): Qwen2MLP(
#           (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
#           (up_proj): Linear(in_features=896, out_features=4864, bias=False)
#           (down_proj): Linear(in_features=4864, out_features=896, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
#         (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
#       )
#     )
#     (norm): Qwen2RMSNorm((896,), eps=1e-06)
#     (rotary_emb): Qwen2RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=896, out_features=151936, bias=False)
# )
# 
# LM Head Structure:
# Linear(in_features=896, out_features=151936, bias=False)
# 
# Hidden size from config: 896
# Vocabulary size from config: 151936

@app.local_entrypoint()
def main():
    modal__train_adapter_ctm.remote()
