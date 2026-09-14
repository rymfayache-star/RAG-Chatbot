import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_DIR = Path(__file__).parent
ABOUT_ME_CANDIDATES = ("Aboutme.txt", "about_me.txt", "AboutMe.txt")
CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTION = (
    "Answer ONLY using this context about RYM. "
    "If the answer is not in the context, say you don’t have that information."
)
EXAMPLE_QUESTIONS = [
    "What are your hobbies?",
    "What languages do you speak?",
    "Where have you traveled?",
]
CUSTOM_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap");

#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stHeader"], [data-testid="stHeaderActionElements"] {
    visibility: hidden !important;
    height: 0 !important;
    max-height: 0 !important;
    display: none !important;
}

.stApp {
    background:
        radial-gradient(1200px 520px at 8% -8%, #dbe7ff 0%, transparent 55%),
        radial-gradient(900px 420px at 100% 0%, #f0e4ff 0%, transparent 48%),
        #f5f7fb;
    color: #0f172a !important;
    font-family: "DM Sans", "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 2.4rem;
    max-width: 820px;
}

.stApp, .stMarkdown, p, span, label, li,
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"],
[data-testid="stExpander"] {
    color: #0f172a !important;
}

h1 {
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    color: #0f172a !important;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 18px;
    padding: 0.35rem 0.2rem;
    box-shadow: 0 10px 28px rgba(28, 41, 82, 0.06);
    margin-bottom: 0.7rem;
}

.welcome-card {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(15, 23, 42, 0.1);
    border-radius: 22px;
    padding: 1.4rem 1.55rem 1.25rem;
    box-shadow: 0 16px 40px rgba(28, 41, 82, 0.08);
    margin: 0.35rem 0 1.4rem;
}

.welcome-card h3 {
    margin: 0 0 0.4rem;
    font-size: 1.25rem;
    color: #0f172a !important;
}

.welcome-card p {
    margin: 0;
    color: #1e293b !important;
    line-height: 1.55;
}

[data-testid="stExpander"] {
    background: #f8f9ff;
    border-radius: 12px;
    border: 1px solid rgba(15, 23, 42, 0.08);
}

[data-testid="stChatInput"] {
    border-radius: 16px;
}

.stSpinner > div {
    color: #4338ca !important;
}

div.stButton > button {
    color: #0f172a !important;
    border-radius: 999px;
    border: 1px solid rgba(15, 23, 42, 0.12);
    background: #ffffff;
}

div.stButton > button:hover {
    border-color: #4338ca;
    color: #312e81 !important;
}

code, pre, .stCode {
    color: #0f172a !important;
}
</style>
"""


def get_openai_api_key() -> str | None:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        secret_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None
    if secret_key is None:
        return None
    secret_key = str(secret_key).strip()
    return secret_key or None


def resolve_about_me_path() -> Path | None:
    for name in ABOUT_ME_CANDIDATES:
        path = APP_DIR / name
        if path.exists():
            return path
    return None


def split_into_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, or fall back to non-empty lines."""
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]
    if len(sentences) <= 1:
        sentences = [line.strip() for line in text.splitlines() if line.strip()]
    return sentences


def overlapping_chunks(
    sentences: list[str],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Build overlapping windows of about 4-5 sentences."""
    if not sentences:
        return []

    size = min(max(size, 1), len(sentences))
    overlap = min(overlap, size - 1) if size > 1 else 0
    stride = size - overlap

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        window = sentences[start : start + size]
        chunks.append("\n".join(window))
        if start + size >= len(sentences):
            break
        start += stride
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def load_knowledge_base(client: OpenAI, about_me_path: Path) -> tuple[list[str], list[list[float]]]:
    text = about_me_path.read_text(encoding="utf-8")
    sentences = split_into_sentences(text)
    chunks = overlapping_chunks(sentences)
    embeddings = embed_texts(client, chunks) if chunks else []
    return chunks, embeddings


def cosine_similarity(query_embedding: list[float], chunk_embeddings: list[list[float]]) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    chunks = np.asarray(chunk_embeddings, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    chunk_norms = np.linalg.norm(chunks, axis=1)
    return (chunks @ query) / (chunk_norms * query_norm + 1e-12)


def retrieve_top_chunks(
    client: OpenAI,
    question: str,
    chunks: list[str],
    embeddings: list[list[float]],
    k: int = TOP_K,
) -> list[str]:
    if not chunks:
        return []
    query_embedding = embed_texts(client, [question])[0]
    scores = cosine_similarity(query_embedding, embeddings)
    k = min(k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[int(i)] for i in top_indices]


def answer_from_context(client: OpenAI, question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks) if context_chunks else "(no context)"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def render_sources(chunks: list[str]) -> None:
    if not chunks:
        return
    with st.expander("Sources"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}**")
            st.code(chunk, language=None)


def show_missing_api_key_error() -> None:
    st.error("OpenAI API key not found.")
    st.markdown(
        """
**How to fix this**

1. Create a `.env` file in the same folder as `rag_chatbot.py`.
2. Add this line: `OPENAI_API_KEY=your_key_here`
3. Or, on Streamlit Cloud, open **App settings → Secrets** and add `OPENAI_API_KEY`.
4. Reload the app.

The app looks for `OPENAI_API_KEY` in the environment first, then in Streamlit secrets.
        """
    )


def show_missing_about_me_error() -> None:
    looked_for = "\n".join(f"- `{APP_DIR / name}`" for name in ABOUT_ME_CANDIDATES)
    st.error("Could not find `about_me.txt` (or `Aboutme.txt`).")
    st.markdown(
        f"""
**How to fix this**

Place a biography file next to `rag_chatbot.py`. The app looked for:

{looked_for}

Add one of those files, then reload the app.
        """
    )


st.set_page_config(page_title="Ask Me Anything About RYM", page_icon="🤖")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

title_col, clear_col = st.columns([4, 1])
with title_col:
    st.title("🤖 Ask Me Anything About RYM")
with clear_col:
    st.write("")
    if st.button("Clear chat", use_container_width=True, key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

api_key = get_openai_api_key()
if not api_key:
    show_missing_api_key_error()
    st.stop()

about_me_path = resolve_about_me_path()
if about_me_path is None:
    show_missing_about_me_error()
    st.stop()

client = OpenAI(api_key=api_key)

if "chunks" not in st.session_state or "embeddings" not in st.session_state:
    with st.spinner("Preparing Rym's knowledge base..."):
        try:
            chunks, embeddings = load_knowledge_base(client, about_me_path)
        except Exception as exc:
            st.error(f"Could not read or index `{about_me_path.name}`.")
            st.markdown(
                f"""
**How to fix this**

Make sure `{about_me_path}` exists, is readable, and that your OpenAI API key is valid.

Details: `{exc}`
                """
            )
            st.stop()
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Hey, welcome 👋</h3>
            <p>
                I’m Rym’s personal Q&amp;A assistant. Ask about her work, family,
                languages, hobbies, or travels — I’ll answer from her profile,
                and you can peek at the source chunks under every reply.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Try an example")
example_cols = st.columns(len(EXAMPLE_QUESTIONS))
example_prompt = None
for i, question in enumerate(EXAMPLE_QUESTIONS):
    if example_cols[i].button(question, use_container_width=True, key=f"example_{i}"):
        example_prompt = question

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

typed_prompt = st.chat_input("Ask me anything about Rym...")
prompt = example_prompt or typed_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_chunks: list[str] = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context_chunks = retrieve_top_chunks(
                    client,
                    prompt,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                )
                reply = answer_from_context(client, prompt, context_chunks)
            except Exception as exc:
                reply = f"Sorry, I could not answer that right now: {exc}"
        st.markdown(reply)
        render_sources(context_chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": context_chunks}
    )
