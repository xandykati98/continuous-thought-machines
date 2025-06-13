import unittest
import torch
import os
import sys

# Adjust path to import from tasks.adapter_llm
# This assumes the test is run from the workspace root or tasks/adapter_llm directory
# For more robust path handling, consider project structure and PYTHONPATH

# Get the directory of the current test file
test_dir = os.path.dirname(os.path.abspath(__file__))
# Get the workspace root (assuming tasks/adapter_llm/inference_test.py structure)
workspace_root = os.path.abspath(os.path.join(test_dir, '../..')) 
sys.path.insert(0, workspace_root)

from tasks.adapter_llm.inference import run_ctm_inference, CTMAdapter, DEFAULT_CTM_CONFIG_FOR_INFERENCE
from transformers import AutoModelForCausalLM

class TestCTMInference(unittest.TestCase):

    def setUp(self):
        self.base_model_name = "Qwen/Qwen2.5-0.5B-Instruct" # Use a small, accessible model for testing
        self.dummy_checkpoint_dir = "dummy_checkpoints_test"
        self.dummy_checkpoint_path = os.path.join(self.dummy_checkpoint_dir, "dummy_adapter_test_checkpoint.pth")
        self.ctm_config = DEFAULT_CTM_CONFIG_FOR_INFERENCE.copy()

        # Create a dummy checkpoint if it doesn't exist
        if not os.path.exists(self.dummy_checkpoint_path):
            print(f"Creating a dummy checkpoint at {self.dummy_checkpoint_path} for testing.")
            os.makedirs(self.dummy_checkpoint_dir, exist_ok=True)
            try:
                # Need LLM hidden dim to initialize adapter
                # This might download the model if not cached, which can be slow for a unit test.
                # Consider mocking this part if speed is critical and model download is an issue.
                _dummy_base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
                _llm_hidden_dim = _dummy_base_model.config.hidden_size
                del _dummy_base_model
                
                dummy_adapter = CTMAdapter(llm_hidden_dim=_llm_hidden_dim, ctm_config=self.ctm_config)
                torch.save(dummy_adapter.state_dict(), self.dummy_checkpoint_path)
                print("Dummy test checkpoint created successfully.")
            except Exception as e:
                print(f"Could not create dummy base model or adapter for test checkpoint: {e}")
                # If dummy checkpoint creation fails, the test that relies on it might fail
                # or the inference script's warning about missing checkpoint will be triggered.
                pass # Allow test to proceed and potentially fail on load if path doesn't exist

    def tearDown(self):
        # Clean up the dummy checkpoint
        if os.path.exists(self.dummy_checkpoint_path):
            os.remove(self.dummy_checkpoint_path)
            print(f"Removed dummy checkpoint: {self.dummy_checkpoint_path}")
        if os.path.exists(self.dummy_checkpoint_dir) and not os.listdir(self.dummy_checkpoint_dir):
            os.rmdir(self.dummy_checkpoint_dir)
            print(f"Removed dummy checkpoint directory: {self.dummy_checkpoint_dir}")

    def test_run_inference_produces_output(self):
        prompt = "Hello, CTM!"
        
        # Ensure the dummy checkpoint path is used, or a real one if provided for manual testing
        # For automated tests, relying on setUp creating the dummy is typical.
        checkpoint_to_use = self.dummy_checkpoint_path
        if not os.path.exists(checkpoint_to_use):
            print(f"Test might show warnings as checkpoint {checkpoint_to_use} doesn't exist and couldn't be created.")
            # The inference script will print a warning and use an initialized adapter.

        generated_text = run_ctm_inference(
            base_model_name=self.base_model_name,
            adapter_checkpoint_path=checkpoint_to_use, 
            ctm_config=self.ctm_config,
            prompt_text=prompt,
            max_new_tokens=10 # Keep generation short for test speed
        )
        self.assertIsInstance(generated_text, str)
        # We expect some output, even if it's not coherent without a trained adapter
        # If the adapter is purely initialized, it should still run.
        # The exact output is non-deterministic and depends on many factors.
        # A simple check is that it's not None and is a string.
        # Checking for non-empty might be too strict if EOS is generated immediately.
        print(f"Test Inference Output: {generated_text}")

if __name__ == '__main__':
    unittest.main() 