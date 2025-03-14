import streamlit as st
import requests
import json

# Page config
st.set_page_config(page_title="Research Assistant", page_icon="🔍")

# Title
st.title("🔍 Research Assistant")
st.markdown("An AI-powered research tool using Perplexity API")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    # Get API key from secrets or allow user override
    default_api_key = st.secrets.get("PERPLEXITY_API_KEY", "")
    api_key = st.text_input("Perplexity API Key", 
                          value=default_api_key,
                          type="password", 
                          help="Using default API key from secrets. You can override it here.")
    model = st.selectbox("Model", ["sonar-reasoning-pro", "sonar-pro"], index=0)

# Function to call Perplexity API directly
def call_perplexity_api(prompt, api_key, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an elite research assistant with exceptional analytical abilities."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("What would you like to research?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Check if API key is provided
    if not api_key:
        with st.chat_message("assistant"):
            st.write("Please enter your Perplexity API key in the sidebar to continue.")
        st.session_state.messages.append({"role": "assistant", "content": "Please enter your Perplexity API key in the sidebar to continue."})
    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    answer = call_perplexity_api(prompt, api_key, model)
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
