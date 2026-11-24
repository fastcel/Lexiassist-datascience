import ollama
from chunker import Chunk

SYSTEM_PROMPT = """
You are LexiAssist, an AI legal assistant that explains Australian law in simple English. 
- Clearly identify the jurisdiction of the law being discussed.
- Never invent section numbers, legislation names, or citations.
- Cite only the numbered sources provided in the context.
- If the retrieved sources do not support an answer, explicitly say so.
- Write in plain English using short paragraphs.
- For lists, use numbered items (1., 2., 3.).
- Do not use Markdown formatting such as asterisks or backslashes.
- Mention section numbers only when explicitly supported by the provided source.
- If the question concerns state or territory law, specify the jurisdiction.

Rules:
- ONLY use the provided context
- Cite sources like [1], [2]
- Never hallucinate
- Be concise
- End with:
  Disclaimer: This is general legal information, not legal advice.
"""

def build_context(retrieved):
    context_parts = []
    citations = []

    for i, item in enumerate(retrieved, start=1):

        chunk: Chunk = item["chunk"]
        score = item["score"]

        
        context_parts.append(
            f"[{i}] Source: {chunk.citation or 'Unknown source'}\n"
            f"Jurisdiction: {chunk.jurisdiction or 'Unknown'}\n"
            f"Passage: {chunk.text}"
        )


        citations.append({
            "index": i,
            "citation": chunk.citation,
            "jurisdiction": chunk.jurisdiction,
            "url": chunk.url,
            "rerank_score": round(score, 4),
            "excerpt": chunk.text[:200]
        })

    return "\n\n".join(context_parts), citations


def generate_answer(query, retrieved):

    if not retrieved:
        return {
            "answer": "No relevant documents found.",
            "citations": [],
            "model": "llama3",
            "query": query,
        }

    context, citations = build_context(retrieved)

    prompt = f"""
Question:
{query}

Context:
{context}

Answer using ONLY the context above.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.message.content

    return {
        "answer": answer,
        "citations": citations,
        "model": "llama3",
        "query": query,
    }