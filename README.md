---
title: Mini Me
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: mit
short_description: A RAG Gen AI chatbot
---


# Mini Me 💬

Mini Me is a Retrieval-Augmented Generation (RAG) GenAI chatbot built with Gradio and OpenAI, designed for technical interviews and recruiter screening. It uses OpenAI embeddings and a custom context index to answer questions based on your LinkedIn or other PDF profile.

## Features
- RAG chatbot with OpenAI GPT-4o-mini
- Customizable relevance threshold for answer acceptance
- PDF-to-embeddings pipeline for context creation
- Gradio web UI with system message, max tokens, temperature, and relevance threshold controls
- Self-aware and context-based answering for interview scenarios

## Setup

1. **Clone the repository**
2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
3. **Set your OpenAI API key:**
	```bash
	export OPENAI_API_KEY="sk-..."
	```
4. **Prepare your context index:**
	- Use `pdf2embeddings.py` to convert a PDF (e.g., LinkedIn profile) into embeddings and metadata:
	  ```bash
	  python pdf2embeddings.py your_profile.pdf
	  ```
	- This will create `index/embeddings.npy` and `index/chunks.json`.

## Running the App

```bash
python app.py
```
Or, for a shareable public link:
```bash
python app.py  # and set share=True in demo.launch()
```

## Usage
- Ask technical or personal questions in the chat UI.
- Adjust the relevance threshold slider to control how strict the context matching is.
- System message, max tokens, and temperature are also configurable in the UI.

## License
MIT

---
Built with [Gradio](https://gradio.app) and [OpenAI](https://platform.openai.com/).
