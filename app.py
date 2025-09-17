import os
import re
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

import gradio as gr
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- CONFIG ----------
MODEL_NAME = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"                 # for query embedding at runtime
TOP_K = 4
MAX_INPUT_WORDS = 50
MAX_OUTPUT_WORDS = 200
MAX_OUTPUT_TOKENS = 300
TEMPERATURE_DEFAULT = 0.0
RELEVANCE_THRESHOLD = 0.18  # dot-product threshold (tuneable)

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "index"
EMBS_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "chunks.json"

# Load embeddings and metadata required for retrieval
if not INDEX_DIR.exists():
    raise RuntimeError(f"Index directory not found at {INDEX_DIR}. Please ensure embeddings and metadata are generated.")

try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load metadata from {META_PATH}: {e}")

try:
    embs = np.load(EMBS_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load embeddings from {EMBS_PATH}: {e}")

# Ensure embeddings are 2D and normalized row-wise for dot-product / cosine similarity
if embs.ndim == 1:
    embs = embs.reshape(1, -1)
embs = normalize(embs, axis=1)

# ---------- utilities ----------
def words_count(s: str) -> int:
    return len(s.strip().split()) if s and s.strip() else 0


def embed_query_openai(query: str):
    """Embed query using OpenAI embedding API. Returns normalized numpy array."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in env. Set it or modify code to provide key.")
    resp = client.embeddings.create(model=EMB_MODEL, input=query)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def retrieve_top_k(qvec, k=TOP_K):
    """Dot product similarity (assumes normalized vectors). Returns list of {'id','text','score'}"""
    scores = embs.dot(qvec)
    idx = np.argsort(scores)[::-1][:k]
    results = []
    for i in idx:
        results.append({"id": meta[i].get("id", str(i)), "text": meta[i].get("text", "")[:1600], "score": float(scores[i])})
    return results


def build_system_prompt():
    # Recruiter screening mode: Only answer if the topic is present in the CONTEXT (from LinkedIn embeddings). Otherwise, say 'I am not sure.'
    return (
        "You are Raj, a distinguished software engineer. "
        "You are answering a recruiter in a first-round screening. "
        "You MUST ONLY answer questions if the topic is present in the provided CONTEXT (from LinkedIn profile/embeddings). "
        "If the topic is not present in CONTEXT, reply exactly: 'I am not sure.' "
        "Be concise and technical as a senior software engineer. "
    )


def compose_context_prompt(question, retrieved):
    ctx = "\n\n".join([f"[id={r['id']}]\n{r['text']}" for r in retrieved])
    prompt = (
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {question}\n\n"
        "Only answer if the question is about something present in the CONTEXT above. "
        "If not, say exactly: I am not sure."
    )
    return prompt


QUOTE_PATTERN = re.compile(r'\[source=(?P<id>[^\]]+)\]\"(?P<quote>.+?)\"', flags=re.DOTALL)

def verify_response_quotes(answer_text: str, retrieved):
    """Verify that any quoted snippets in the answer exist as substrings in the corresponding retrieved chunk text.
       Returns True if at least one valid quote found and all quoted snippets match; False otherwise.
    """
    matches = list(QUOTE_PATTERN.finditer(answer_text))
    if not matches:
        # No explicit quotes -> treat as unsupported
        return False
    retrieved_map = {str(r["id"]): r["text"] for r in retrieved}
    for m in matches:
        cid = m.group("id").strip()
        quote = m.group("quote").strip()
        target = retrieved_map.get(cid, "")
        if not quote or quote not in target:
            return False
    return True


# ---------- OpenAI call ----------
def call_openai_chat(system_prompt, user_prompt, max_tokens, temperature, model_name=MODEL_NAME):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in env. Set it before running the app.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    try:
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = resp["choices"][0]["message"]["content"].strip()
    # usage may be an attribute or a dict entry
    usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else {})
    print(text)
    return text, usage

# ---------- Gradio app logic ----------
def handle_question(user_message, chat_history, system_message, max_tokens_slider, temperature_slider):
    """
    chat_history: list of (user, bot) tuples as provided by gr.Chatbot state.
    """
    # Enforce input length
    if words_count(user_message) > MAX_INPUT_WORDS:
        return chat_history + [(user_message, "Input too long. Please keep it under 50 words.")], ""

    # Embed query (OpenAI embeddings)
    try:
        qvec = embed_query_openai(user_message)
    except Exception as e:
        return chat_history + [(user_message, f"Embedding failed: {e}")], ""

    # Retrieve
    retrieved = retrieve_top_k(qvec, k=TOP_K)
    top_score = retrieved[0]["score"] if (retrieved and len(retrieved) > 0) else 0.0

    # Build prompts
    system_prompt = build_system_prompt() + ("\n\nSystem Note: " + system_message if system_message else "")
    user_prompt = compose_context_prompt(user_message, retrieved)

    # Call generator
    try:
        gen_text, usage = call_openai_chat(system_prompt, user_prompt, max_tokens_slider, temperature_slider)
    except Exception as e:
        return chat_history + [(user_message, f"Generation failed: {e}")], json.dumps(retrieved, indent=2)

    # Verify quoted snippets
    ok = verify_response_quotes(gen_text, retrieved)
    if not ok:
        # accept generated answer if top retrieved chunk is sufficiently similar
        if top_score >= RELEVANCE_THRESHOLD:
            # enforce output word limit
            words = gen_text.strip().split()
            if len(words) > MAX_OUTPUT_WORDS:
                gen_text = " ".join(words[:MAX_OUTPUT_WORDS]) + " ..."
            final = gen_text
            debug_info = f"top_score={top_score:.4f} >= threshold {RELEVANCE_THRESHOLD:.2f}; accepted generated answer without explicit quote."
        else:
            final = "I am not sure."
            debug_info = f"top_score={top_score:.4f} < threshold {RELEVANCE_THRESHOLD:.2f}; verification failed."
    else:
        # enforce output word limit
        words = gen_text.strip().split()
        if len(words) > MAX_OUTPUT_WORDS:
            gen_text = " ".join(words[:MAX_OUTPUT_WORDS]) + " ..."
        final = gen_text
        debug_info = f"verified quotes present; top_score={top_score:.4f}."

    # append to history and return updated chat and debug (retrieved + debug_info)
    display = {"retrieved": retrieved, "debug": debug_info, "usage": usage}
    display_debug = json.dumps(display, default=str, indent=2)
    return chat_history + [(user_message, final)], display_debug



# --- Gradio ChatInterface integration with OpenAI logic ---
def gradio_respond(message, history, system_message, max_tokens, temperature, top_p):
    """
    Gradio ChatInterface handler using OpenAI chat completion and retrieval logic.
    history: list of dicts with 'role' and 'content'.
    """
    # Convert history to (user, bot) tuples for handle_question
    chat_history = []
    for turn in history:
        if turn["role"] == "user":
            user_msg = turn["content"]
        elif turn["role"] == "assistant":
            bot_msg = turn["content"]
            chat_history.append((user_msg, bot_msg))
    # Call handle_question for retrieval-augmented QA
    updated_history, debug_info = handle_question(
        message,
        chat_history,
        system_message,
        max_tokens,
        temperature,
    )
    # Only return the latest bot response
    return updated_history[-1][1]


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

chatbot = gr.ChatInterface(
    gradio_respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=300, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Mini-me: OpenAI-powered RAG Chatbot")
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
