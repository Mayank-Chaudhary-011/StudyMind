import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_output(query, retriever, top_k=5):
    results = retriever.retrieve(query, top_k)

    context = "\n".join([doc["document"] for doc in results]) if results else ""

    if not context:
        return "I couldn't find relevant information in your study material."

    prompt = f"""Use the given context to generate the answer for the query.
Context: {context}
Query: {query}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content