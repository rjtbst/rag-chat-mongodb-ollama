import os
from urllib import response
import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = "llama3"

def chat_with_llama(messages):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": messages,
            "stream": False
        })
        response.raise_for_status()
        data = response.json()
        return data.get('message', {}).get('content', 'No response content.')
    
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


# 🧠 Maintain full conversation history
conversation = []

print("💬 Chatbot started! (Type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Goodbye!")
        break

    # Add user message
    conversation.append({"role": "user", "content": user_input})

    # Call the model with full history
    reply = chat_with_llama(conversation)

    # Add assistant's reply to history
    conversation.append({"role": "assistant", "content": reply})

    # Show reply
    print("AI:", reply)

