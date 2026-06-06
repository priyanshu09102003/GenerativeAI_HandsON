import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")

st.markdown("""
<style>
    /* Hide default streamlit header */
    #MainMenu, footer { visibility: hidden; }

    /* Clean chat container */
    .stChatMessage { border-radius: 12px; margin-bottom: 4px; }

    /* Thinking indicator */
    .thinking {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #888;
        font-size: 0.85rem;
        padding: 8px 0;
    }
    .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #888;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
        40%            { transform: translateY(-6px); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ── Header 
st.title("🤖 Simple Chatbot")
st.caption("Powered by Gemini 2.5 Flash · LangChain")
st.divider()

# ── Model + state
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []      # {"role": ..., "content": ...}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # flat list passed to model

# ── Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
        <div class="thinking">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            &nbsp;Thinking...
        </div>
    """, unsafe_allow_html=True)

    # Build history + invoke (mirrors your CLI version exactly)
    st.session_state.chat_history.append(user_input)
    result = model.invoke(st.session_state.chat_history)
    response = result.content
    st.session_state.chat_history.append(response)

    # Clear thinking, show response
    thinking_placeholder.empty()
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)