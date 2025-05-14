import os
import gc
import shutil

os.makedirs(os.environ.get('HF_HOME', '~/.cache'), exist_ok=True)

import torch
from dataclasses import dataclass
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from handlers.llms.base_llm import BaseLLM

def format_prompt(query: str, tokenizer):
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False                
    )

@dataclass(frozen=True)
class HuggingFaceLLM(BaseLLM):
    """
    HuggingFaceLLM is a class for handling generate calls to HuggingFace models.
    Extension of the abstract BaseLLM class.
    """


    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: Any
    
    @staticmethod
    def initialize_llm_client(
        model_name: str,
        auth_token: str,
        torch_dtype: Optional[str | torch.dtype] = torch.bfloat16,
        device: Optional[str | torch.device] = None,
    ):
        """
        Initialize the model, tokenizer and device.
        """

        try:
            if device is not None:
                _device = torch.device(device)
            else:
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if _device.type == 'cuda' else None,
                token=auth_token
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                token=auth_token
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        
            return model, tokenizer, _device
        except Exception as e:
            print(f"Error initializing model and tokenizer for {model_name}: {e}")
            return None, None, None
        
    def format_prompt(self, query: str):
        messages = [{"role": "user", "content": query}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False                
        )
        
    def query(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the Hugging Face model.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The generated text (response) or an error message.
        """

        try:
            if self.model is None or self.tokenizer is None:
                return "Model or tokenizer not initialized."
            
            inputs = self.tokenizer(format_prompt(query_text, self.tokenizer), return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
            reply_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response_text = self.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

            return response_text
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message

    def delete(self):
        """
        Delete the model and tokenizer to free up memory.
        """
        try:
            # Clear CUDA cache if using GPU
            torch.cuda.empty_cache()

            # Clear any remaining references
            gc.collect()
        except Exception as e:
            print(f"Error during deletion of model and tokenizer: {e}")