
import re
import ollama
import numpy as np
from collections import Counter


# --------------------------------
# TEXT PREPROCESSING
# --------------------------------

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "as", "at", "from", "that",
    "this", "these", "those", "it", "its", "be",
    "what", "which", "who", "how", "why", "when",
    "explain", "describe", "define", "give", "me",
    "about", "using", "only", "pdf"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        word for word in words
        if word not in STOPWORDS
    ]


# --------------------------------
# CREATE PAGE-AWARE TEXT CHUNKS
# --------------------------------

def create_chunks(pages, chunk_size=800, overlap=120):

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


# --------------------------------
# GENERATE EMBEDDINGS
# --------------------------------

def generate_embeddings(chunks):

    all_embeddings = []

    batch_size = 8

    for i in range(0, len(chunks), batch_size):

        batch = [
            chunk["text"]
            for chunk in chunks[i:i + batch_size]
        ]

        response = ollama.embed(
            model="nomic-embed-text",
            input=batch
        )

        if isinstance(response, dict):
            batch_embeddings = response["embeddings"]
        else:
            batch_embeddings = response.embeddings

        all_embeddings.extend(batch_embeddings)

    return np.array(
        all_embeddings,
        dtype=np.float32
    )


# --------------------------------
# COSINE SIMILARITY
# --------------------------------

def semantic_search(question, embeddings):

    response = ollama.embed(
        model="nomic-embed-text",
        input=question
    )

    if isinstance(response, dict):
        query_embedding = response["embeddings"][0]
    else:
        query_embedding = response.embeddings[0]

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    query_norm = np.linalg.norm(query_embedding)

    scores = np.dot(
        embeddings,
        query_embedding
    ) / (
        embedding_norms * query_norm + 1e-10
    )

    return scores


# --------------------------------
# BM25 KEYWORD SEARCH
# --------------------------------

def bm25_search(question, chunks, k1=1.5, b=0.75):

    query_words = tokenize(question)

    documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if not query_words:
        return np.zeros(len(chunks), dtype=np.float32)

    document_count = len(documents)

    document_lengths = [
        len(doc) for doc in documents
    ]

    average_length = (
        sum(document_lengths) / document_count
        if document_count else 0
    )

    document_frequency = Counter()

    for document in documents:

        for word in set(document):
            document_frequency[word] += 1

    scores = []

    for document, length in zip(
        documents,
        document_lengths
    ):

        word_counts = Counter(document)

        score = 0.0

        for word in query_words:

            frequency = word_counts[word]

            if frequency == 0:
                continue

            df = document_frequency[word]

            idf = np.log(
                1 + (
                    document_count - df + 0.5
                ) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * length / (
                    average_length + 1e-10
                )
            )

            score += (
                idf
                * frequency
                * (k1 + 1)
                / denominator
            )

        scores.append(score)

    return np.array(
        scores,
        dtype=np.float32
    )


# --------------------------------
# NORMALIZE SCORES
# --------------------------------

def normalize_scores(scores):

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum - minimum < 1e-10:
        return np.zeros_like(scores)

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# --------------------------------
# HYBRID DOCUMENT SEARCH
# --------------------------------

def search_documents(
    question,
    chunks,
    embeddings,
    top_k=4
):

    if not chunks or embeddings is None:
        return []

    # Semantic similarity

    semantic_scores = semantic_search(
        question,
        embeddings
    )

    # Keyword relevance

    keyword_scores = bm25_search(
        question,
        chunks
    )

    # Normalize both scores

    semantic_normalized = normalize_scores(
        semantic_scores
    )

    keyword_normalized = normalize_scores(
        keyword_scores
    )

    # Combine semantic and keyword scores

    if np.max(keyword_scores) > 0:

        final_scores = (
            0.55 * semantic_normalized
            + 0.45 * keyword_normalized
        )

    else:

        final_scores = semantic_normalized

    # Select the most relevant chunks

    best_indices = np.argsort(
        final_scores
    )[::-1][:top_k]

    results = []

    for index in best_indices:

        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(final_scores[index]),
            "semantic_score": float(
                semantic_normalized[index]
            ),
            "keyword_score": float(
                keyword_normalized[index]
            )
        })

    return results
