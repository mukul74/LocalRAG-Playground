# 🧠 LocalRAG-Playground

A simple and extensible playground to **analyze different locally running LLMs via RAG pipelines**, powered by **[Ollama](https://ollama.com/)**.  
This repo helps study:

- 🕒 **Inference latency**
- 📄 **Answer quality variations across models**
- 🧩 **Effect of chunking and retrieval**
- 🔄 **Consistency of responses to same questions**

---

## 🔍 What This Project Does

This project explores:

- Running **multiple LLMs locally** using Ollama (e.g., `llama3.2`, `deepseek`, etc.)
- Comparing **response quality** and **speed** of each model
- Studying different **chunking strategies** and **retrieval setups**
- Logging and analyzing model behavior for **same set of queries**

---

## 🛠️ Stack

- **Ollama** – for serving local LLMs
- **LangChain / LlamaIndex** – for building RAG pipelines
- **Pandas / Matplotlib** – for results tracking and visualization (optional)
- **Python** – the core logic

---

## 📦 Setup

```bash
# Start your preferred LLMs
ollama run llama3
ollama run deepseek

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main_comparision.py
```

This will:
- Load the models
- Run predefined questions through the RAG setup
- Log inference time + responses
- Save comparisons to CSV or JSON

---

## 📊 Output Example

| Question         | Model   | Inference Time (s) | Response Snippet         |
|------------------|---------|---------------------|---------------------------|
| What is RAG?     | llama3  | 1.42                | "RAG stands for..."       |
| What is RAG?     | mistral | 1.08                | "RAG is a method to..."   |

---

## 📌 TODO

- [ ] Add answer grading metrics
- [ ] Add support for more models
- [ ] Improve visualizations

---
