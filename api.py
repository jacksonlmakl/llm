from flask import Flask, jsonify, request
from agent import Agent

app = Flask(__name__)

global AGENT
AGENT=Agent()

@app.route('/chat', methods=['GET'])
def chat():
    prompt = request.args.get('prompt', '')
    rag = request.args.get('prompt', False)
    stream = request.args.get('stream', False)

    response=AGENT.chat(prompt,rag=rag,stream=stream)
    if not prompt or prompt=='':
        return jsonify({
            'error': 'Missing required parameter: prompt'
        }), 400
    
    # Create response object
    response = {
        'role' : 'assistant',
        'content': response,
    }
    if len(AGENT.model.messages)>10:
        AGENT.model.messages=AGENT.model.messages[1:11]
    return jsonify(response)

@app.route('/clear', methods=['GET'])
def clear():
    response={'success':AGENT.model.clear()}
    
    return jsonify(response)

@app.route('/reset', methods=['GET'])
def reset():
    global AGENT
    AGENT=Agent()
    
    response={'success':True}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)