import accelerate
import torch
from transformers import pipeline
from huggingface_hub import login
import logging

class Model:
    def __init__(self, token, model_id="meta-llama/Llama-3.2-3B-Instruct",messages=[]):
        logging.disable(logging.CRITICAL)  # Disable all logging
        login(token)
        self.pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )
        self.messages=messages
    def chat(self, prompt, save=False,max_new_tokens=256):
        _messages=self.messages+[{"role":"user","content":prompt}],
        outputs = self.pipe(
            _messages,
            max_new_tokens=max_new_tokens,
        )
        try:
            if save:
                self.messages=outputs[0][-1]['generated_text']
            return outputs[0][-1]['generated_text'][-1]['content']
        except:
            if save:
                self.messages=outputs[0]['generated_text']
            return outputs[0]['generated_text'][-1]['content']
    def clear(self):
        self.messages=[]
        return True