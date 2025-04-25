from flask import Flask, request, jsonify
import os
import accelerate
import torch
from transformers import pipeline
from huggingface_hub import login


class Model:
    def __init__(self, token, model_id="meta-llama/Llama-3.2-3B-Instruct", messages=[]):
        login(token)
        if torch.cuda.is_available():
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            dtype = torch.float32
        else:
            dtype = torch.float32

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.messages = messages

    def chat(self, prompt, save=False, max_new_tokens=256):
        _messages = self.messages + [{"role": "user", "content": prompt}]
        output = self.pipe(_messages, max_new_tokens=max_new_tokens)
        response = output[0]["generated_text"]

        if save:
            self.messages.append({"role": "assistant", "content": response})

        return response

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Extract required parameters
    model_id = data.get('model_id')
    token = data.get('token')
    messages = data.get('messages', [])
    prompt = data.get('prompt', '')
    stream = data.get('stream', False)
    
    # Validate required parameters
    if not all([model_id, token]):
        return jsonify({
            'error': 'Missing required parameters. Please provide model_id and token.'
        }), 400
    
    try:
        # Initialize model
        model = Model(token, model_id, messages)
        
        # Generate response
        response = model.chat(prompt, stream)
        
        return jsonify({
            'response': str(response)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)