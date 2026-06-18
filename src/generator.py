from llm import generate_completion

def generate_output(query, retriever, top_k=5):
    results = retriever.retrieve(query, top_k)

    context = "\n".join([doc["document"] for doc in results]) if results else ""

    if not context:
        return "I couldn't find relevant information in your study material."

    prompt = f"""Use the given context to generate the answer for the query.
Context: {context}
Query: {query}"""

    return generate_completion(prompt, temperature=0.1, max_tokens=1024)