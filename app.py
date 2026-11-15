
import streamlit as st
import ollama
import io
import hashlib

from pypdf import PdfReader

from rag_utils import (
    create_chunks,
    generate_embeddings,
    search_documents
)


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Advanced AI PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced AI PDF Chatbot")
st.caption("Powered by Llama 3.2, Ollama and RAG Technology")


# --------------------------------
# SESSION STATE
# --------------------------------

defaults = {
    "messages": [],
    "chunks": [],
    "embeddings": None,
    "file_hash": None,
    "page_count": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------
# PDF UPLOAD
# --------------------------------

st.sidebar.header("📄 Upload your PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(pdf_bytes).hexdigest()

    if current_hash != st.session_state.file_hash:

        try:
            with st.spinner("Processing your PDF..."):

                reader = PdfReader(io.BytesIO(pdf_bytes))

                pages = []

                for page_number, page in enumerate(
                    reader.pages, start=1
                ):

                    page_text = page.extract_text() or ""

                    if page_text.strip():

                        pages.append({
                            "page": page_number,
                            "text": page_text
                        })

                if not pages:
                    st.error(
                        "No extractable text found in this PDF. "
                        "It may contain scanned images."
                    )
                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:
                    st.error("No text chunks were created.")
                    st.stop()

                embeddings = generate_embeddings(chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.file_hash = current_hash
                st.session_state.page_count = len(reader.pages)
                st.session_state.messages = []

            st.sidebar.success("PDF processed successfully!")

        except Exception as e:
            st.sidebar.error(f"PDF Processing Error: {e}")
            st.stop()

    else:
        st.sidebar.success("PDF is ready!")

    st.sidebar.write(
        "Total PDF pages:",
        st.session_state.page_count
    )

    st.sidebar.write(
        "Total text chunks:",
        len(st.session_state.chunks)
    )


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------
# USER QUESTION
# --------------------------------

prompt = st.chat_input("Ask a question about your PDF...")


if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:

            if not st.session_state.chunks:

                answer = "Please upload a PDF document first."
                st.warning(answer)

            else:

                with st.spinner(
                    "Searching PDF and generating answer..."
                ):

                    # Retrieve the most relevant chunks
                    results = search_documents(
                        prompt,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        top_k=6
                    )

                    if not results:

                        answer = (
                            "The answer was not found in the "
                            "uploaded PDF."
                        )

                        st.warning(answer)

                    else:

                        # Remove duplicate passages
                        unique_results = []
                        seen_text = set()

                        for result in results:

                            text = result["text"].strip()

                            if text and text not in seen_text:
                                unique_results.append(result)
                                seen_text.add(text)

                        # Build retrieved PDF context
                        context_parts = []

                        for index, result in enumerate(
                            unique_results, start=1
                        ):

                            context_parts.append(
                                f"""
--- PDF SOURCE {index} ---
PAGE NUMBER: {result['page']}

PDF TEXT:
{result['text']}
"""
                            )

                        context = "\n".join(context_parts)

                        # Strict PDF-only instructions
                        system_prompt = f"""
You are a PDF question-answering assistant.

Your only source of information is the PDF text
provided below.

STRICT RULES:

1. Answer only using facts explicitly present
   in the supplied PDF text.

2. Never use your general knowledge or outside
   information.

3. Never invent definitions, examples, facts,
   quotations or explanations.

4. Do not assume missing information.

5. If the answer is not clearly available in the
   supplied PDF text, respond exactly:

   "The answer was not found in the retrieved
   sections of the uploaded PDF."

6. If only some parts of the question are answered
   by the PDF, explain only those parts.

7. Use simple and clear language.

8. Do not create page numbers. The application
   displays retrieved PDF pages separately.

9. Do not repeat the question unnecessarily.

10. Do not add examples unless they appear in
    the supplied PDF text.

PDF CONTENT:

{context}
"""

                        # Send only the current question
                        # to avoid unrelated chat history
                        conversation = [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]

                        # Generate answer
                        response = ollama.chat(
                            model="llama3.2:1b",
                            messages=conversation,
                            options={
                                "temperature": 0,
                                "num_ctx": 8192
                            }
                        )

                        answer = response["message"]["content"].strip()

                        st.markdown(answer)

                        # --------------------------------
                        # SOURCE REFERENCES
                        # --------------------------------

                        source_pages = sorted(
                            set(
                                result["page"]
                                for result in unique_results
                            )
                        )

                        st.markdown("### 📚 Retrieved Source Pages")

                        st.write(
                            "The following pages supplied the "
                            "retrieved text used for this answer:"
                        )

                        for page_number in source_pages:

                            st.markdown(
                                f"📄 **PDF Page {page_number}**"
                            )

                            with st.expander(
                                f"View extracted text from Page {page_number}"
                            ):

                                for result in unique_results:

                                    if result["page"] == page_number:
                                        st.write(result["text"])

                        answer += (
                            "\n\n**Retrieved Source Pages:** "
                            + ", ".join(
                                str(page) for page in source_pages
                            )
                        )

            # Save response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(f"AI Error: {e}")


# --------------------------------
# CLEAR CHAT
# --------------------------------

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []
    st.rerun()
