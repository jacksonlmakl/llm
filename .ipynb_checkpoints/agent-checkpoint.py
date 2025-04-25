from model import Model
from rag import RAG
import logging
import warnings

class Agent:
    def __init__(self, token, model_id="meta-llama/Llama-3.2-1B-Instruct", messages=None):
        logging.disable(logging.CRITICAL)  # Disable all logging
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        if type(messages) == list:
            self.messages = messages
        else:
            self.messages = [
                {"role": "system", "content": "You are a helpful, smart assistant good at parsing text, language, and summarizing."},
                {"role": "system", "content": "You will be given a prompt from the user, and some information that may be relevant. Answer the user prompt accuratley using relevant information."}
            ]
            
        self.model=Model(token,model_id,self.messages)
    def chat(self,prompt,rag=True,stream=True):
        logging.disable(logging.CRITICAL)  # Disable all logging
        if rag:
            enriched_prompt=RAG(prompt)
            new_p=f"{prompt} \n\n---- Potentially Relevant Information From RAG: \n\n\n {enriched_prompt}"
            outputs=self.model.chat(new_p,stream)
        else:
            outputs=self.model.chat(f"{prompt}",stream)
        return outputs