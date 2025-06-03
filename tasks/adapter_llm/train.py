# Load model directly
from pathlib import Path
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("transformers", "datasets", "torch", "huggingface-hub", "wandb")

app = modal.App(name="adapter-ctm-training-app", image=image)

MODEL_DIR = Path("/models")

volume = modal.Volume.from_name("adapter-ctm", create_if_missing=True)
@app.function(gpu="A10G", image=image, timeout=3600*4, volumes={MODEL_DIR: volume})
def modal__train_adapter_ctm():
    
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
    wandb.login(key="")
    try:
        login(token="") 
        print("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        print(f"Hugging Face Hub login failed: {e}. Dataset loading might fail if it's private.")

    # Load dataset
    try:
        ds = load_dataset("PrimeIntellect/real-world-swe-problems", split="train") # Using train split
        print(f"Dataset loaded. Number of examples: {len(ds)}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Exiting due to dataset loading failure.")
        exit()


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    # Add padding token if it doesn't exist. For Causal LMs, often pad_token = eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.eos_token_id # This line caused an error if model not defined yet
        print("Set tokenizer.pad_token to tokenizer.eos_token")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")

    # Define a mock adapter
    class Adapter(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super(Adapter, self).__init__()
            self.down_project = nn.Linear(input_dim, bottleneck_dim)
            self.relu = nn.ReLU()
            self.up_project = nn.Linear(bottleneck_dim, input_dim)

        def forward(self, x):
            residual = x
            x = self.down_project(x)
            x = self.relu(x)
            x = self.up_project(x)
            return x + residual

    # Hidden size from user comment/model config
    hidden_size = model.config.hidden_size # Get hidden size from the loaded model's config
    print(f"Using hidden_size: {hidden_size}")

    # Initialize W&B
    # Generate a unique run ID for this training session
    run_id = f"run_Qwen_Adapter_0.5B_Instruct_{int(time.time())}" 
    wandb.init(project="adapter_llm_training", name=run_id, config={
        "learning_rate": 1e-4, # Example, adjust as needed
        "epochs": 5, # Example, adjust as needed
        "batch_size": 3, # Example, adjust as needed
        "model_name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "adapter_bottleneck": 1024,
        "dataset": "PrimeIntellect/real-world-swe-problems",
        "run_id": run_id,
        "wandb_api_key": "5c0d2d6b1fcad21af4e0cc3894c119285c4ddae5"
    })
    # Instantiate the adapter
    adapter = Adapter(input_dim=hidden_size, bottleneck_dim=wandb.config.adapter_bottleneck) # bottleneck_dim can be tuned


    # Training Loop Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    adapter.to(device)

    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False
    print("Froze base model parameters.")

    # Adapter parameters are trainable by default as it's a new module.
    # Ensure adapter parameters require grad (should be true by default for new nn.Module)
    for param in adapter.parameters():
        param.requires_grad = True
    print("Adapter parameters are trainable.")


    # Data preprocessing function
    def preprocess_function(examples):
        # Concatenate prompt and gold_standard_solution for Causal LM training
        # The model learns to predict the solution given the prompt.
        inputs = [prompt + tokenizer.eos_token + solution + tokenizer.eos_token 
                for prompt, solution in zip(examples["prompt"], examples["gold_standard_solution"])]
        
        # Tokenize
        # max_length should be chosen based on model's capacity and typical sequence lengths in dataset
        # Padding will be handled by the DataCollatorForLanguageModeling
        model_inputs = tokenizer(inputs, max_length=32768, padding="do_not_pad", truncation=True)
        
        # DataCollatorForLanguageModeling will create labels by copying input_ids and handling padding for labels with -100
        # So, we don't need to create model_inputs["labels"] here.
        return model_inputs

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

    print("Starting dataset preprocessing for training data...")
    if train_ds and len(train_ds) > 0:
        tokenized_train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
        print("Training dataset preprocessing finished.")
    else:
        print("Skipping training dataset preprocessing as train_ds is empty or None.")
        tokenized_train_ds = None

    # Function to generate and log a sample
    def generate_and_log_sample(model, adapter, tokenizer, device, epoch, run_id, epoch_completion, current_loss, test_ds, train_ds, max_new_tokens=30000, base_filename="generation_sample.json"):
        # global test_ds, train_ds # Ensure we're using the globally defined datasets

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

        # Select a random sample
        sample_index = random.randint(0, len(sample_source_ds) - 1)
        raw_sample = sample_source_ds[sample_index]
        raw_prompt_text = raw_sample["prompt"]
        gold_solution = raw_sample["gold_standard_solution"]
        
        # Tokenize only the prompt part for generation input
        prompt_ids = tokenizer(raw_prompt_text + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=32768).input_ids.to(device)

        if prompt_ids is None: # Should not happen if a sample was selected
            print("Skipping sample generation as no prompt is available after selection.")
            return

        model.eval()
        adapter.eval()
        generated_ids = prompt_ids.clone() # Start with the prompt

        with torch.no_grad():
            for _ in range(max_new_tokens):
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
        
        # Save to the main generation_sample.json (overwrite)
        try:
            with open(base_filename, 'w', encoding='utf-8') as f:
                json.dump(sample_output, f, indent=4, ensure_ascii=False)
            print(f"Saved latest sample generation to {base_filename}")
        except Exception as e:
            print(f"Error saving sample generation to {base_filename}: {e}")

        # Save detailed record to a separate directory
        log_dir = "generation_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        detailed_filename = os.path.join(log_dir, f"log_epoch{epoch}_run{run_id}_time{time.strftime('%Y%m%d-%H%M%S')}.json")
        try:
            with open(detailed_filename, 'w', encoding='utf-8') as f:
                json.dump(sample_output, f, indent=4, ensure_ascii=False)
            print(f"Saved detailed sample generation to {detailed_filename}")
        except Exception as e:
            print(f"Error saving detailed sample generation to {detailed_filename}: {e}")
        
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


    # Optimizer
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
        #generate_and_log_sample(model, adapter, tokenizer, device, epoch=epoch, run_id=run_id, epoch_completion=0.0, current_loss=None, test_ds=test_ds, train_ds=train_ds)


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
    #     (embed_tokens): Embedding(151936, 1536)
    #     (layers): ModuleList(
    #       (0-27): 28 x Qwen2DecoderLayer(
    #         (self_attn): Qwen2Attention(
    #           (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
    #           (k_proj): Linear(in_features=1536, out_features=256, bias=True)
    #           (v_proj): Linear(in_features=1536, out_features=256, bias=True)
    #           (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
    #         )
    #         (mlp): Qwen2MLP(
    #           (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
    #           (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
    #           (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
    #           (act_fn): SiLU()
    #         )
    #         (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
    #         (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
    #       )
    #     )
    #     (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    #     (rotary_emb): Qwen2RotaryEmbedding()
    #   )
    #   (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
    # )
    # 
    # LM Head Structure:
    # Linear(in_features=1536, out_features=151936, bias=False)
    # 
    # Hidden size from config: 1536
    # Vocabulary size from config: 151936

@app.local_entrypoint()
def main():
    modal__train_adapter_ctm.remote()
