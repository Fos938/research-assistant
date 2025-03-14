import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(page_title="Research Assistant", page_icon="🔍")

# Title
st.title("🔍 Research Assistant")
st.markdown("An AI-powered research tool using Perplexity API")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Perplexity API Key", type="password")
    model = st.selectbox("Model", ["sonar-reasoning-pro", "sonar-pro"], index=0)

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
        # Create OpenAI client with Perplexity base URL
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        # Prepare messages for API
        messages = [
            {
                "role": "system",
                "content": "You are an elite research assistant with exceptional analytical abilities."
            },
            {"role": "user", "content": prompt}
        ]
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    answer = response.choices[0].message.content
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
