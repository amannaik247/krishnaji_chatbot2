import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face API key
api_key = os.getenv(HUGGINGFACE_API_KEY)
model_name = "amannaik/talkwithsrikrishna"  # You can use any public model, e.g., "gpt2" or "distilgpt2"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define a chat prompt template
prompt_template = """
The following is a conversation with Lord Krishna. Lord Krishna is helpful and understanding.He will always tell me the absolute truth.

User: {user_input}
Lord Krishna:"""

st.title("Chat with Lord Krishna")

# Function to generate response via Hugging Face API
def generate_response(input_text):
    # Combine all chat history for context
    context = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in st.session_state.messages])
    
    # Use the prompt template
    prompt = prompt_template.format(user_input=input_text)
    full_context = context + "\n" + prompt

    # API request to Hugging Face
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": full_context,
        "parameters": {
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    response = requests.post(f"https://api-inference.huggingface.co/models/{model_name}", headers=headers, json=payload)

    # Extract response text
    generated_text = response.json()[0]['generated_text']
    bot_response = generated_text.split("Bot:")[-1].strip()
    return bot_response

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        # Generate response
        bot_response = generate_response(user_input)

        # Update chat history
        st.session_state.messages.append({"user": user_input, "bot": bot_response})

        # Display chat history
        for chat in st.session_state.messages:
            st.write(f"**You:** {chat['user']}")
            st.write(f"**Bot:** {chat['bot']}")
            st.write("---")

        # Clear input box after sending
        st.session_state.user_input = ""
