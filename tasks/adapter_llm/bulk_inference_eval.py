import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json
import time
import re
from pathlib import Path
import modal
from datasets import load_dataset
from huggingface_hub import login
import openai
from collections import defaultdict

# Modal App Setup
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers", 
    "torch", 
    "huggingface-hub", 
    "accelerate",
    "datasets",
    "openai"
).add_local_python_source("models")

app = modal.App(name="bulk-inference-eval-app")

MODEL_DIR = Path("/models")
volume = modal.Volume.from_name("adapter-ctm", create_if_missing=True)

# Import CTM with fallback
try:
    from models.ctm import ContinuousThoughtMachine
except ImportError as e:
    print(f"Error importing ContinuousThoughtMachine: {e}")
    # Fallback for local execution
    if 'ContinuousThoughtMachine' not in globals():
        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.abspath(os.path.join(current_file_dir, '../..'))
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
            from models.ctm import ContinuousThoughtMachine
            print("Successfully imported ContinuousThoughtMachine via sys.path modification.")
        except ImportError as e_fallback:
            raise ImportError(f"Failed to import ContinuousThoughtMachine: {e_fallback}")

# CTM Adapter Class (same as in inference.py)
class CTMAdapter(nn.Module):
    def __init__(self, llm_hidden_dim, ctm_config):
        super().__init__()
        self.llm_hidden_dim = llm_hidden_dim
        
        self.ctm_core = ContinuousThoughtMachine(
            iterations=ctm_config['ctm_iterations'],
            d_model=ctm_config['ctm_d_model'],
            d_input=ctm_config['ctm_internal_attn_dim'],
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
        
        self.refinement = nn.Linear(llm_hidden_dim, llm_hidden_dim)
        
    def forward(self, x_llm):
        batch_size, seq_len, _ = x_llm.shape
        residual = x_llm
        
        ctm_predictions, _, _ = self.ctm_core(x_llm)
        processed_features = ctm_predictions[:, :, -1]
        processed_features = processed_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        delta_h = self.refinement(processed_features)
        return residual + delta_h

# Default CTM Config
DEFAULT_CTM_CONFIG = {
    "ctm_bottleneck_dim": 896,
    "ctm_d_model": 896,
    "ctm_internal_attn_dim": 896,
    "ctm_iterations": 32,
    "ctm_heads": 8,
    "ctm_n_synch_out": 512,
    "ctm_n_synch_action": 512,
    "ctm_synapse_depth": 2,
    "ctm_memory_length": 8,
    "ctm_deep_nlms": True,
    "ctm_memory_hidden_dims": 32,
    "ctm_do_layernorm_nlm": False,
    "ctm_dropout": 0.1,
    "ctm_dropout_nlm": 0.1,
    "ctm_neuron_select_type": 'random-pairing',
    "ctm_n_random_pairing_self": 4,
    "ctm_backbone_type": "token-processing",
    "ctm_positional_embedding_type": "sequence-rotational",
}

def generate_text(model, adapter, tokenizer, prompt_text, device, max_new_tokens=5000, temperature=0.7, top_p=0.9, use_adapter=True):
    """Generate text using model with or without adapter"""
    print(f"Generating text (adapter={use_adapter}, max_tokens={max_new_tokens})...")
    
    model.eval()
    if adapter:
        adapter.eval()
    
    # Prepare inputs
    prompt_ids = tokenizer.encode(prompt_text + tokenizer.eos_token, return_tensors="pt", truncation=True).to(device)
    generated_ids = prompt_ids.clone()
    tokens_generated = 0

    with torch.no_grad():
        for i in range(max_new_tokens):
            if i % 50 == 0:
                print(f"  Generated {i}/{max_new_tokens} tokens...")
                
            if use_adapter and adapter:
                outputs = model(input_ids=generated_ids, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                adapted_hidden_states = adapter(last_hidden_states)
                next_token_logits = model.lm_head(adapted_hidden_states)[:, -1, :]
            else:
                outputs = model(input_ids=generated_ids)
                next_token_logits = outputs.logits[:, -1, :]

            # Sampling logic
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
            tokens_generated += 1

            if next_token_id.item() == tokenizer.eos_token_id:
                print("  EOS token generated. Stopping.")
                break
        
    generated_sequence = generated_ids[0]
    generated_text_only = tokenizer.decode(generated_sequence[prompt_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"Generated text length: {len(generated_text_only)} characters, {tokens_generated} tokens")
    return generated_text_only, tokens_generated

def evaluate_with_openai(prompt, raw_response, adapter_response, gold_solution, openai_api_key):
    """Use OpenAI to evaluate which response is better"""
    print("Evaluating responses with OpenAI...")
    
    client = openai.OpenAI(api_key=openai_api_key, base_url="https://api.deepseek.com")
    
    evaluation_prompt = f"""You are an expert code reviewer evaluating two AI model responses to a coding problem.

ORIGINAL PROMPT:
{prompt}

GOLD STANDARD SOLUTION:
{gold_solution}

RAW MODEL RESPONSE:
{raw_response}

ADAPTER MODEL RESPONSE:  
{adapter_response}

Please evaluate both responses and provide your assessment using exactly these XML tags:

<closest_to_gold>RAW|ADAPTER</closest_to_gold>
<adapter_response_score>0-100</adapter_response_score>
<raw_response_score>0-100</raw_response_score>

Consider:
1. Correctness and functionality compared to the gold solution
2. Code quality, readability, and best practices
3. Completeness of the solution

Provide scores from 0-100 where:
- 90-100: Excellent, very close to gold solution
- 70-89: Good, mostly correct with minor issues
- 50-69: Fair, partially correct but missing key elements
- 30-49: Poor, significant issues but some correct elements
- 0-29: Very poor, mostly incorrect or incomplete

Choose which response is closest to the gold solution (RAW or ADAPTER).
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer. Always respond with the requested XML tags."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        evaluation_text = response.choices[0].message.content
        print(f"OpenAI evaluation received: {len(evaluation_text)} characters")
        return evaluation_text
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def detect_mode_collapse(raw_response, adapter_response, openai_api_key):
    """Use OpenAI to detect if the adapter response shows mode collapse"""
    print("Checking for mode collapse with OpenAI...")
    
    client = openai.OpenAI(api_key=openai_api_key, base_url="https://api.deepseek.com")
    
    collapse_prompt = f"""You are an expert at detecting mode collapse in language models. Mode collapse typically manifests as:
1. Repetitive text or patterns
2. Incoherent rambling that doesn't follow the input structure
3. Excessive verbosity without meaningful content
4. Getting stuck in loops or repetitive phrases

RAW MODEL RESPONSE (for comparison):
{raw_response}

ADAPTER MODEL RESPONSE (much longer):
{adapter_response}

The adapter response is significantly longer than the raw response. Please analyze if this indicates mode collapse.

Respond with exactly this XML tag:
<collapse>YES|NO</collapse>

Choose YES if the adapter response shows clear signs of mode collapse (repetition, incoherence, meaningless verbosity).
Choose NO if the adapter response is just more detailed/comprehensive but still coherent and meaningful.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at detecting mode collapse in AI models. Always respond with the requested XML tag."},
                {"role": "user", "content": collapse_prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        collapse_text = response.choices[0].message.content
        print(f"Mode collapse check received: {collapse_text}")
        
        # Parse the collapse tag
        collapse_match = re.search(r'<collapse>(YES|NO)</collapse>', collapse_text)
        if collapse_match:
            collapse_result = collapse_match.group(1)
            print(f"Mode collapse detected: {collapse_result}")
            return collapse_result == "YES"
        else:
            print("Failed to parse collapse response, assuming NO collapse")
            return False
        
    except Exception as e:
        print(f"Error calling OpenAI API for collapse detection: {e}")
        return False

def parse_evaluation(evaluation_text):
    """Parse the XML tags from OpenAI evaluation"""
    if not evaluation_text:
        return None, None, None
    
    try:
        # Extract closest_to_gold
        closest_match = re.search(r'<closest_to_gold>(RAW|ADAPTER)</closest_to_gold>', evaluation_text)
        closest_to_gold = closest_match.group(1) if closest_match else None
        
        # Extract adapter score
        adapter_score_match = re.search(r'<adapter_response_score>(\d+)</adapter_response_score>', evaluation_text)
        adapter_score = int(adapter_score_match.group(1)) if adapter_score_match else None
        
        # Extract raw score
        raw_score_match = re.search(r'<raw_response_score>(\d+)</raw_response_score>', evaluation_text)
        raw_score = int(raw_score_match.group(1)) if raw_score_match else None
        
        print(f"Parsed evaluation - Closest: {closest_to_gold}, Adapter: {adapter_score}, Raw: {raw_score}")
        return closest_to_gold, adapter_score, raw_score
        
    except Exception as e:
        print(f"Error parsing evaluation: {e}")
        return None, None, None

@app.function(gpu="A10G", image=image, volumes={str(MODEL_DIR): volume}, timeout=7200)
def modal_bulk_inference_eval(
    run_id: str,
    epoch_number: int,
    openai_api_key: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    checkpoint_filename: str = "adapter_checkpoint.pth",
    num_samples: int = 100,
    max_new_tokens: int = 5000
):
    """Run bulk inference evaluation on Modal"""
    print(f"Starting bulk inference evaluation for run_id: {run_id}, epoch: {epoch_number}")
    print(f"Will evaluate {num_samples} samples")
    
    # Login to Hugging Face
    try:
        login(token="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")
        print("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        print(f"Hugging Face Hub login failed: {e}")
    
    # Load dataset
    print("Loading PrimeIntellect dataset...")
    try:
        ds = load_dataset("PrimeIntellect/real-world-swe-problems", split="train")
        print(f"Dataset loaded. Total examples: {len(ds)}")
        
        # Take first num_samples
        if len(ds) > num_samples:
            ds = ds.select(range(num_samples))
            print(f"Selected first {num_samples} samples")
        else:
            print(f"Dataset has only {len(ds)} samples, using all")
            
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    # Setup device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer.pad_token to tokenizer.eos_token")
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is not None and base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set base_model.config.pad_token_id to {tokenizer.pad_token_id}")
    
    base_model.to(device)
    llm_hidden_dim = base_model.config.hidden_size
    print(f"Model hidden dimension: {llm_hidden_dim}")
    
    # Load adapter
    print("Loading CTM adapter...")
    adapter = CTMAdapter(llm_hidden_dim=llm_hidden_dim, ctm_config=DEFAULT_CTM_CONFIG)
    
    # Construct checkpoint path
    checkpoint_path_on_volume = MODEL_DIR / "checkpoints" / run_id / f"epoch_{epoch_number}" / checkpoint_filename
    print(f"Looking for adapter checkpoint at: {checkpoint_path_on_volume}")
    
    if os.path.exists(checkpoint_path_on_volume):
        adapter.load_state_dict(torch.load(checkpoint_path_on_volume, map_location=device))
        print("Loaded CTM adapter weights successfully")
    else:
        print(f"Warning: Adapter checkpoint not found at {checkpoint_path_on_volume}")
        print("Continuing with initialized adapter (this will likely produce poor results)")
    
    adapter.to(device)
    
    # Initialize counters
    results = {
        'total_samples': 0,
        'successful_evaluations': 0,
        'mode_collapses': 0,
        'skipped_due_to_collapse': 0,
        'judge_llm_errors': 0,
        'network_errors': 0,
        'parsing_errors': 0,
        'closest_to_gold_counts': defaultdict(int),
        'adapter_scores': [],
        'raw_scores': [],
        'evaluation_details': [],
        'collapse_details': [],
        'error_details': [],
        'run_id': run_id,
        'epoch_number': epoch_number,
        'base_model_name': base_model_name,
        'max_new_tokens': max_new_tokens,
        'num_samples_planned': num_samples
    }
    
    print("\n" + "="*80)
    print("🚀 STARTING BULK INFERENCE AND EVALUATION")
    print("="*80)
    print("💡 Enhanced Features:")
    print("  • Real-time stats after every sample")
    print("  • Comprehensive error tracking and recovery")
    print("  • Intermediate results saved every 5 samples")
    print("  • Mode collapse detection with fallback")
    print("  • Network error resilience")
    print("="*80)
    
    # Process each sample
    for i, sample in enumerate(ds):
        print(f"\n--- PROCESSING SAMPLE {i+1}/{len(ds)} ---")
        results['total_samples'] += 1
        
        prompt = sample["prompt"]
        gold_solution = sample["gold_standard_solution"]
        
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Gold solution length: {len(gold_solution)} characters")
        
        # Generate with raw model
        print("\n1. Generating with RAW model...")
        raw_response, raw_tokens = generate_text(
            base_model, None, tokenizer, prompt, device, 
            max_new_tokens=max_new_tokens, use_adapter=False
        )
        
        # Calculate dynamic max tokens for adapter (2.5x raw response)
        adapter_max_tokens = int(raw_tokens * 2.5)
        print(f"\nRAW model generated {raw_tokens} tokens. Setting adapter max to {adapter_max_tokens} tokens (2.5x)")
        
        # Generate with adapter
        print("\n2. Generating with ADAPTER model...")
        adapter_response, adapter_tokens = generate_text(
            base_model, adapter, tokenizer, prompt, device,
            max_new_tokens=adapter_max_tokens, use_adapter=True
        )
        
        print(f"\nRaw response: {len(raw_response)} characters, {raw_tokens} tokens")
        print(f"Adapter response: {len(adapter_response)} characters, {adapter_tokens} tokens")
        
        # Check for mode collapse if adapter reached the token limit (indicating it wanted to generate more)
        adapter_to_raw_ratio = adapter_tokens / max(raw_tokens, 1)  # Avoid division by zero
        print(f"Adapter/Raw token ratio: {adapter_to_raw_ratio:.2f}")
        
        is_collapsed = False
        collapse_check_failed = False
        
        # Check for collapse if adapter hit the limit (meaning it would have gone over 2.5x if allowed)
        if adapter_tokens >= adapter_max_tokens * 0.95:  # 95% of limit indicates likely truncation
            print(f"\n⚠️ Adapter hit token limit ({adapter_tokens}/{adapter_max_tokens}) - checking for mode collapse...")
            
            try:
                is_collapsed = detect_mode_collapse(raw_response, adapter_response, openai_api_key)
                
                if is_collapsed:
                    print(f"🚨 MODE COLLAPSE DETECTED for sample {i+1}! Skipping evaluation.")
                    results['mode_collapses'] += 1
                    results['skipped_due_to_collapse'] += 1
                    
                    # Store collapse details
                    collapse_detail = {
                        'sample_index': i,
                        'token_ratio': adapter_to_raw_ratio,
                        'raw_tokens': raw_tokens,
                        'adapter_tokens': adapter_tokens,
                        'adapter_max_tokens': adapter_max_tokens,
                        'raw_length_chars': len(raw_response),
                        'adapter_length_chars': len(adapter_response),
                        'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        'collapse_detection_method': 'openai_judge'
                    }
                    results['collapse_details'].append(collapse_detail)
                    
                    continue  # Skip to next sample
                else:
                    print(f"✅ No mode collapse detected despite hitting token limit.")
                    
            except Exception as e:
                print(f"❌ Error checking for mode collapse: {e}")
                print(f"⚠️ Proceeding with evaluation despite collapse check failure...")
                collapse_check_failed = True
                results['network_errors'] += 1  # Count collapse detection errors as network errors
                
                # Store error details
                error_detail = {
                    'sample_index': i,
                    'error_type': 'collapse_detection_error',
                    'error_message': str(e),
                    'token_ratio': adapter_to_raw_ratio,
                    'adapter_tokens': adapter_tokens,
                    'adapter_max_tokens': adapter_max_tokens,
                    'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                }
                results['error_details'].append(error_detail)
        
        # Evaluate with OpenAI
        print("\n3. Evaluating with OpenAI...")
        try:
            evaluation_text = evaluate_with_openai(
                prompt, raw_response, adapter_response, gold_solution, openai_api_key
            )
            
            if evaluation_text:
                closest_to_gold, adapter_score, raw_score = parse_evaluation(evaluation_text)
                
                if closest_to_gold and adapter_score is not None and raw_score is not None:
                    results['successful_evaluations'] += 1
                    results['closest_to_gold_counts'][closest_to_gold] += 1
                    results['adapter_scores'].append(adapter_score)
                    results['raw_scores'].append(raw_score)
                    
                    # Store detailed results
                    sample_result = {
                        'sample_index': i,
                        'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        'raw_response': raw_response[:200] + "..." if len(raw_response) > 200 else raw_response,
                        'adapter_response': adapter_response[:200] + "..." if len(adapter_response) > 200 else adapter_response,
                        'closest_to_gold': closest_to_gold,
                        'adapter_score': adapter_score,
                        'raw_score': raw_score,
                        'evaluation_text': evaluation_text
                    }
                    results['evaluation_details'].append(sample_result)
                    
                    print(f"✅ Sample {i+1} evaluated successfully")
                    print(f"   Closest to gold: {closest_to_gold}")
                    print(f"   Adapter score: {adapter_score}")
                    print(f"   Raw score: {raw_score}")
                else:
                    print(f"❌ Failed to parse evaluation for sample {i+1}")
                    results['parsing_errors'] += 1
                    
                    # Store error details
                    error_detail = {
                        'sample_index': i,
                        'error_type': 'parsing_error',
                        'evaluation_text': evaluation_text[:500] + "..." if len(evaluation_text) > 500 else evaluation_text,
                        'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    }
                    results['error_details'].append(error_detail)
            else:
                print(f"❌ Failed to get evaluation from OpenAI for sample {i+1}")
                results['judge_llm_errors'] += 1
                
                # Store error details
                error_detail = {
                    'sample_index': i,
                    'error_type': 'judge_llm_error',
                    'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                }
                results['error_details'].append(error_detail)
                
        except Exception as e:
            print(f"❌ Network/API error for sample {i+1}: {e}")
            results['network_errors'] += 1
            
            # Store error details
            error_detail = {
                'sample_index': i,
                'error_type': 'network_error',
                'error_message': str(e),
                'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
            }
            results['error_details'].append(error_detail)
        
        # Print cumulative statistics after EVERY sample
        print(f"\n--- CUMULATIVE STATS AFTER SAMPLE {i+1}/{len(ds)} ---")
        print(f"📊 Overall Progress:")
        print(f"  • Total processed: {results['total_samples']}")
        print(f"  • Successful evaluations: {results['successful_evaluations']} ({(results['successful_evaluations']/results['total_samples']*100):.1f}%)")
        print(f"  • Mode collapses: {results['mode_collapses']} ({(results['mode_collapses']/results['total_samples']*100):.1f}%)")
        print(f"  • Skipped due to collapse: {results['skipped_due_to_collapse']}")
        
        print(f"📈 Error Breakdown:")
        print(f"  • Judge LLM errors: {results['judge_llm_errors']}")
        print(f"  • Network errors: {results['network_errors']}")
        print(f"  • Parsing errors: {results['parsing_errors']}")
        
        if results['successful_evaluations'] > 0:
            print(f"🏆 Performance Metrics:")
            total_evals = results['successful_evaluations']
            for model_type, count in results['closest_to_gold_counts'].items():
                percentage = (count / total_evals) * 100
                print(f"  • {model_type} wins: {count}/{total_evals} ({percentage:.1f}%)")
            
            avg_adapter = sum(results['adapter_scores']) / len(results['adapter_scores'])
            avg_raw = sum(results['raw_scores']) / len(results['raw_scores'])
            print(f"  • Average adapter score: {avg_adapter:.1f}")
            print(f"  • Average raw score: {avg_raw:.1f}")
            print(f"  • Score difference: {avg_adapter - avg_raw:+.1f}")
        else:
            print(f"⚠️  No successful evaluations yet.")
        
        # Save intermediate results every 5 samples (in case of crash)
        if (i + 1) % 5 == 0:
            try:
                results_dir = MODEL_DIR / "bulk_inference_results"
                os.makedirs(results_dir, exist_ok=True)
                
                intermediate_filename = f"intermediate_bulk_eval_{run_id}_epoch_{epoch_number}_{int(time.time())}.json"
                intermediate_path = results_dir / intermediate_filename
                
                # Convert defaultdict to regular dict for JSON serialization
                results_for_json = dict(results)
                results_for_json['closest_to_gold_counts'] = dict(results['closest_to_gold_counts'])
                results_for_json['samples_processed_so_far'] = i + 1
                results_for_json['total_planned_samples'] = len(ds)
                
                if results['total_samples'] > 0:
                    results_for_json['collapse_rate_percentage'] = (results['mode_collapses'] / results['total_samples']) * 100
                else:
                    results_for_json['collapse_rate_percentage'] = 0
                
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(results_for_json, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Intermediate results saved to: {intermediate_filename}")
                volume.commit()
                
            except Exception as save_e:
                print(f"⚠️  Failed to save intermediate results: {save_e}")
        
        print("─" * 60)
    
    # Final statistics
    print("\n" + "="*80)
    print("🏁 FINAL RESULTS")
    print("="*80)
    
    print(f"📋 Processing Summary:")
    print(f"  • Total samples processed: {results['total_samples']}")
    print(f"  • Successful evaluations: {results['successful_evaluations']} ({(results['successful_evaluations']/results['total_samples']*100):.1f}%)")
    success_rate = (results['successful_evaluations'] / results['total_samples']) * 100 if results['total_samples'] > 0 else 0
    
    print(f"\n📊 Error Analysis:")
    print(f"  • Mode collapses detected: {results['mode_collapses']} ({(results['mode_collapses']/results['total_samples']*100):.1f}%)")
    print(f"  • Samples skipped due to collapse: {results['skipped_due_to_collapse']}")
    print(f"  • Judge LLM errors: {results['judge_llm_errors']} ({(results['judge_llm_errors']/results['total_samples']*100):.1f}%)")
    print(f"  • Network/API errors: {results['network_errors']} ({(results['network_errors']/results['total_samples']*100):.1f}%)")
    print(f"  • Parsing errors: {results['parsing_errors']} ({(results['parsing_errors']/results['total_samples']*100):.1f}%)")
    
    total_errors = results['mode_collapses'] + results['judge_llm_errors'] + results['network_errors'] + results['parsing_errors']
    print(f"  • Total error rate: {total_errors}/{results['total_samples']} ({(total_errors/results['total_samples']*100):.1f}%)")
    
    if results['mode_collapses'] > 0:
        print(f"\n🔄 Mode Collapse Details:")
        collapse_rate = (results['mode_collapses'] / results['total_samples']) * 100
        print(f"  • Collapse rate: {collapse_rate:.1f}%")
        if results['collapse_details']:
            print(f"  • Average token ratio in collapsed samples: {sum(c['token_ratio'] for c in results['collapse_details'])/len(results['collapse_details']):.2f}")
            print(f"  • Average raw tokens in collapsed samples: {sum(c['raw_tokens'] for c in results['collapse_details'])/len(results['collapse_details']):.1f}")
            print(f"  • Average adapter tokens in collapsed samples: {sum(c['adapter_tokens'] for c in results['collapse_details'])/len(results['collapse_details']):.1f}")
    
    if results['successful_evaluations'] > 0:
        print(f"\n🏆 Performance Metrics:")
        for model_type, count in results['closest_to_gold_counts'].items():
            percentage = (count / results['successful_evaluations']) * 100
            print(f"  • {model_type} wins: {count}/{results['successful_evaluations']} ({percentage:.1f}%)")
        
        avg_adapter = sum(results['adapter_scores']) / len(results['adapter_scores'])
        avg_raw = sum(results['raw_scores']) / len(results['raw_scores'])
        print(f"  • Average adapter score: {avg_adapter:.1f}")
        print(f"  • Average raw score: {avg_raw:.1f}")
        print(f"  • Score difference (adapter - raw): {avg_adapter - avg_raw:+.1f}")
        
        print(f"  • Score ranges:")
        print(f"    - Adapter: {min(results['adapter_scores'])}-{max(results['adapter_scores'])}")
        print(f"    - Raw: {min(results['raw_scores'])}-{max(results['raw_scores'])}")
        
        # Additional metrics
        adapter_wins = results['closest_to_gold_counts'].get('ADAPTER', 0)
        raw_wins = results['closest_to_gold_counts'].get('RAW', 0)
        if adapter_wins + raw_wins > 0:
            win_rate = (adapter_wins / (adapter_wins + raw_wins)) * 100
            print(f"  • Adapter win rate: {win_rate:.1f}%")
    else:
        print(f"\n⚠️  No successful evaluations completed!")
        
    # Error breakdown by sample indices (helpful for debugging)
    if results['error_details']:
        print(f"\n🐛 Error Sample Indices:")
        error_by_type = defaultdict(list)
        for error in results['error_details']:
            error_by_type[error['error_type']].append(error['sample_index'])
        
        for error_type, indices in error_by_type.items():
            print(f"  • {error_type}: samples {sorted(indices)}")
    
    print(f"\n💡 Data Quality Assessment:")
    if success_rate >= 80:
        print(f"  ✅ Excellent: {success_rate:.1f}% success rate - results are highly reliable")
    elif success_rate >= 60:
        print(f"  ✅ Good: {success_rate:.1f}% success rate - results are reliable")
    elif success_rate >= 40:
        print(f"  ⚠️  Fair: {success_rate:.1f}% success rate - results have moderate reliability")
    elif success_rate >= 20:
        print(f"  ⚠️  Poor: {success_rate:.1f}% success rate - results have low reliability")
    else:
        print(f"  ❌ Very Poor: {success_rate:.1f}% success rate - results may not be reliable")
    
    # Save results to volume
    results_dir = MODEL_DIR / "bulk_inference_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_filename = f"bulk_eval_{run_id}_epoch_{epoch_number}_{int(time.time())}.json"
    results_path = results_dir / results_filename
    
    # Convert defaultdict to regular dict for JSON serialization
    results_for_json = dict(results)
    results_for_json['closest_to_gold_counts'] = dict(results['closest_to_gold_counts'])
    
    # Add collapse rate calculation for JSON
    if results['total_samples'] > 0:
        results_for_json['collapse_rate_percentage'] = (results['mode_collapses'] / results['total_samples']) * 100
    else:
        results_for_json['collapse_rate_percentage'] = 0
    
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")
        
        volume.commit()
        print("Results committed to Modal volume")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results_for_json

@app.local_entrypoint()
def main():
    """Local entrypoint to run bulk inference evaluation"""
    
    # Configuration - UPDATE THESE VALUES
    test_run_id = "run_Qwen_Adapter_0.5B_Instruct__CTM__1749351122"  # Replace with your actual run_id
    test_epoch_number = 1  # Replace with your actual epoch number
    openai_api_key = "sk-2468088d02c84ee783f5c6394315e42f"  # Replace with your OpenAI API key
    
    if openai_api_key == "your_openai_api_key_here":
        print("ERROR: Please set your OpenAI API key in the script!")
        print("Update the 'openai_api_key' variable in the main() function.")
        return
    
    print("="*80)
    print("BULK INFERENCE EVALUATION")
    print("="*80)
    print(f"Run ID: {test_run_id}")
    print(f"Epoch: {test_epoch_number}")
    print(f"Samples to evaluate: 100")
    print("="*80)
    
    # Run the bulk evaluation on Modal
    try:
        results = modal_bulk_inference_eval.remote(
            run_id=test_run_id,
            epoch_number=test_epoch_number,
            openai_api_key=openai_api_key,
            num_samples=100,
            max_new_tokens=5000
        )
        
        if results:
            print("\n✅ Bulk evaluation completed successfully!")
            print("Check the printed output above for detailed results.")
        else:
            print("\n❌ Bulk evaluation failed!")
            
    except Exception as e:
        print(f"\n❌ Error running bulk evaluation: {e}")
        print("Please ensure:")
        print("1. Modal daemon is running and you're logged in")
        print("2. The specified checkpoint exists on the volume")
        print("3. Your OpenAI API key is valid")
        print("4. You have sufficient OpenAI API credits") 