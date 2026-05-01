import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
import pdfplumber
import numpy as np

load_dotenv()

st.set_page_config(
    page_title="Healthcare Document Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Document Assistant")
st.write("Upload a medical PDF and ask questions in simple language.")

st.warning(
    "⚠️ This is an assistive system, not a replacement for doctors. "
    "Always consult a qualified medical professional."
)


def get_api_key():
    return os.getenv("GEMINI_API_KEY", "").strip()


def extract_text_from_pdf(uploaded_file):
    text_pages = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(f"--- Page {page_num} ---\n{page_text}")
        return "\n\n".join(text_pages)
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")


def split_into_chunks(text, chunk_size=600, overlap=120):
    words = text.split()
    chunks = []
    word_chunk = chunk_size // 5
    word_overlap = overlap // 5
    start = 0
    while start < len(words):
        end = min(start + word_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += word_chunk - word_overlap
    return chunks


def embed_texts(texts, api_key):
    client = genai.Client(api_key=api_key)
    embeddings = []

    for text in texts:
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            if not result.embeddings:
                continue
            vector = result.embeddings[0].values
            if vector is not None:
                embeddings.append(list(vector))  # ✅ FIXED: was outside the loop before
        except Exception as e:
            st.warning(f"Skipping chunk due to embedding error: {e}")
            continue

    return np.array(embeddings, dtype="float32")


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def retrieve_relevant_chunks(question, chunks, embeddings, api_key, top_k=4):
    client = genai.Client(api_key=api_key)

    q_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    if not q_result.embeddings:
        raise ValueError("Could not create embedding for the question.")

    q_vector_values = q_result.embeddings[0].values
    if q_vector_values is None:
        raise ValueError("Question embedding is empty.")

    q_vector = np.array([list(q_vector_values)], dtype="float32")
    scores = cosine_similarity(q_vector, embeddings)[0]
    top_k = min(top_k, len(chunks))
    top_indices = scores.argsort()[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    return "\n\n".join(relevant_chunks)


def generate_answer(question, context, api_key):
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a friendly healthcare document assistant.

Your job:
Explain medical documents in very simple language.

Important rules:
1. Do NOT diagnose disease.
2. Do NOT prescribe medicine.
3. Do NOT say "you have this disease".
4. Always ask the user to consult a doctor.
5. Explain like the user is a beginner.

Answer in this format:

**📋 Simple Explanation**
Explain in simple words.

**💡 What This Usually Means**
Explain general meaning only.

**🩺 Questions to Ask Your Doctor**
Give 3-4 questions.

**⚠️ Safety Note**
This is an assistive system, not a replacement for doctors.

Context from document:
{context}

User question:
{question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"


# ── Session state init ──────────────────────────────────────────────────────
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── API key check ───────────────────────────────────────────────────────────
api_key = get_api_key()

if not api_key:
    st.error("""
    🔑 Gemini API Key not found.

    Create a `.env` file and write:

    GEMINI_API_KEY=your_api_key_here

    Get a free key at: https://aistudio.google.com/
    """)
    st.stop()

# ── PDF upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF report", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Reading and processing PDF... (this may take a minute for large files)"):
            try:
                text = extract_text_from_pdf(uploaded_file)

                if not text.strip():
                    st.error("Could not extract text. This PDF may be scanned/image-based.")
                    st.stop()

                chunks = split_into_chunks(text)
                st.info(f"📄 Extracted {len(chunks)} chunks from the PDF. Generating embeddings...")

                embeddings = embed_texts(chunks, api_key)

                if embeddings.shape[0] == 0:
                    st.error("Embedding failed — no vectors were generated. Check your API key.")
                    st.stop()

                st.session_state.pdf_text = text
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.pdf_processed = True

                st.success(f"✅ PDF processed! {embeddings.shape[0]} chunks embedded and ready.")

            except Exception as e:
                st.error(f"Error: {e}")

# ── Q&A interface ───────────────────────────────────────────────────────────
if st.session_state.pdf_processed:
    with st.expander("View extracted PDF text"):
        st.text_area(
            "Extracted Text",
            value=st.session_state.pdf_text,
            height=300
        )

    st.subheader("Ask questions about your report")

    example_questions = [
        "Explain this report in simple words",
        "Are any values abnormal?",
        "What does hemoglobin mean?",
        "What should I ask my doctor?"
    ]

    selected_question = st.selectbox(
        "Choose example question or type your own below",
        [""] + example_questions
    )

    user_question = st.text_input("Your question")
    final_question = user_question if user_question else selected_question

    if st.button("Ask"):
        if final_question:
            with st.spinner("Thinking..."):
                try:
                    context = retrieve_relevant_chunks(
                        final_question,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        api_key
                    )
                    answer = generate_answer(final_question, context, api_key)

                    st.session_state.chat_history.append(
                        {"role": "user", "content": final_question}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")

    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Assistant:** {chat['content']}")
            st.markdown("---")

else:
    st.info("Upload and process a PDF first.")
