# 🧠 Generative AI with LangChain

<div align="center">

A hands-on repository to practice and recap the most fundamental GenAI concepts through **project-based learning**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Google-4285F4?style=flat-square&logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

---

## 📌 Projects

| # | Project | Key Learnings | Link |
|---|---------|---------------|------|
| 01 | **Basic Research Assistant** | Passing prompts to LLMs/Chat Models — static & dynamic | [→ View](https://research-assistant-demo.streamlit.app/) |
| 02 | **Basic Chatbot** | Wiring up UI with Streamlit, getting structured responses, implementing the memory to the LLM, Types of message templates in LangChain | [→ View](https://chatbot-demo-langchain.streamlit.app/) |
| 03 | **Customer Care Support Agent - Feedback Responder** | Learnt the concepts of **Runnables** , **Chaining** and **StructuredOutput (using parsers)** by developing a fundamental feedback-responding agent that can classify customer reviews as **POSITIVE** or **NEGATIVE** and uses **Conditional Chaining**  using runnable to respond to them accordingly| [→ View](https://github.com/priyanshu09102003/GenerativeAI_HandsON/blob/main/Chains/condition_chain.py) |
| 03 | **Joke Machine** | Deep-dived in the concept of **Runnables** and use of different **Chains** by developing a Joke generation machine that writes a joke on a given topic and explains it.| [→ View](https://github.com/priyanshu09102003/GenerativeAI_HandsON/blob/main/Runnables/runnable_passthrough.py) |
| 04 | **Equity Research Analyst - Full RAG Pipeline** | A full-stack **RAG (Retrieval-Augmented Generation) pipeline** that ingests live news articles via URL loaders, chunks them using recursive character text splitters, and converts chunks into semantic vector embeddings using HuggingFace sentence-transformers stored in a FAISS index. At query time, the retriever performs similarity search over the vector store to surface the most relevant context, which is then passed to Gemini 2.5 Flash for grounded, source-backed answers — eliminating hallucinations by anchoring the LLM strictly to retrieved document chunks.

Learned the core concepts of **RAG pipelines**, **vector embeddings**, **document loaders**, and **text splitters**, together in a single pipeline while building the tool.
| [→ View](https://github.com/priyanshu09102003/GenerativeAI_HandsON/blob/main/Runnables/runnable_passthrough.py) |

---


<div align="center">
  <sub>Built while learning · Elementary level · Aspiring GenAI Engineer 🔧</sub>
</div>