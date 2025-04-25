from flask import Flask, request, jsonify, session
from flask_session import Session
from agent import Agent
import uuid

app = Flask(__name__)

# Configure server-side session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = "your_secret_key_here"  # Change this to a secure random key in production
Session(app)

# Store API token (you might want to use environment variables in production)
HF_TOKEN = "***"

# Endpoint to handle chat interactions
@app.route("/chat", methods=["POST"])
def chat():
    # Get or create user session ID
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    
    user_id = session["user_id"]
    
    # Create or retrieve agent for this user session
    if "agent" not in session:
        session["agent"] = Agent(HF_TOKEN)
    
    # Get message from request
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    
    message = data["message"]
    
    # Use the agent to process the message
    response = session["agent"].chat(message)
    
    return jsonify({
        "user_id": user_id,
        "response": response
    })

# Endpoint to get a summary of the conversation
@app.route("/summarize", methods=["GET"])
def summarize():
    if "agent" not in session:
        return jsonify({"error": "No active conversation to summarize"}), 400
    
    summary = session["agent"].summarize()
    
    return jsonify({
        "user_id": session.get("user_id", "unknown"),
        "summary": summary
    })

# Endpoint to get the full message history
@app.route("/history", methods=["GET"])
def history():
    if "agent" not in session:
        return jsonify({"error": "No active conversation"}), 400
    
    # Access the underlying message chain
    messages = session["agent"].model.messages
    
    return jsonify({
        "user_id": session.get("user_id", "unknown"),
        "messages": messages
    })

# Endpoint to create a new session (reset conversation)
@app.route("/new_session", methods=["POST"])
def new_session():
    # Clear the current session
    session.clear()
    
    # Create a new user ID and agent
    session["user_id"] = str(uuid.uuid4())
    session["agent"] = Agent(HF_TOKEN)
    
    return jsonify({
        "user_id": session["user_id"],
        "message": "New session created"
    })

if __name__ == "__main__":
    app.run(debug=True)