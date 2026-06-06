import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 Your Chatbot")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Type a message...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get and show assistant response
    result = model.invoke(user_input)
    response = result.content

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)