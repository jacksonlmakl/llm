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
    def summarize(self):
        self.chat("Please create a detailed summary of our entire conversation and your responses",rag=False)
        summary=self.model.messages[-1]['content']
        self.model.clear()
        self.model.messages=[{"role":"system","content":f"You are a helpful assistant very good at language, parsing, and understanding text, here is a summary of your previous conversation to reference \n\n----- \n\n{summary}"}]