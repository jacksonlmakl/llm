from flask import Flask, jsonify, request
from agent import Agent
from s3 import sync_s3_to_documents

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

@app.route('/s3', methods=['GET'])
def s3():
    aws_access_key = request.args.get('aws_access_key', None)
    aws_secret_key = request.args.get('aws_secret_key', None)
    bucket_name = request.args.get('bucket_name', None)
    print(bucket_name)
    if not bucket_name:
        raise Exception("bucket_name required")
    if not aws_access_key and not aws_secret_key:
        public_bucket=True
    else:
        public_bucket=False
    sync_s3_to_documents(bucket_name, 
                         public_bucket=public_bucket,
                         aws_access_key=aws_access_key, 
                         aws_secret_key=aws_secret_key)
    
    response={'success':True}
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=5000)