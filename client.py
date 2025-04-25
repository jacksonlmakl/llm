import requests
import json

class AgentApiClient:
    """
    A wrapper class for interacting with the Agent Flask API.
    Maintains session cookies to ensure communication with the same agent instance.
    """
    
    def __init__(self, host="http://localhost:5000"):
        """
        Initialize the API client.
        
        Args:
            base_url (str): The base URL of the Flask API
        """
        self.base_url = host
        self.session = requests.Session()
        self.user_id = None
    
    def chat(self, message):
        """
        Send a message to the agent and get the response.
        
        Args:
            message (str): The message to send to the agent
            
        Returns:
            dict: The response from the agent
        """
        url = f"{self.base_url}/chat"
        payload = {"message": message}
        response = self.session.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.user_id = data.get("user_id")
            return data.get("response")
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def summarize(self):
        """
        Get a summary of the conversation.
        
        Returns:
            str: The conversation summary
        """
        url = f"{self.base_url}/summarize"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json().get("summary")
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def get_history(self):
        """
        Get the full conversation history.
        
        Returns:
            list: The complete message history
        """
        url = f"{self.base_url}/history"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json().get("messages")
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def new_session(self):
        """
        Create a new conversation session.
        
        Returns:
            str: The new user ID
        """
        url = f"{self.base_url}/new_session"
        response = self.session.post(url)
        
        if response.status_code == 200:
            data = response.json()
            self.user_id = data.get("user_id")
            return self.user_id
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def get_user_id(self):
        """
        Get the current user ID.
        
        Returns:
            str: The user ID
        """
        return self.user_id