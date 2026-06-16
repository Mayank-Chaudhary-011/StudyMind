import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

QUIZ_PROMPT = """
You are an expert teacher. Based on the study material below, generate {num_questions} multiple choice questions.

Rules:
- Each question must have exactly 4 options labeled A, B, C, D
- Only one option is correct
- Questions should test genuine understanding
- Focus on: {topic}

Study Material:
{context}

Respond ONLY with a valid JSON array like this (no extra text, no markdown):
[
  {{
    "question": "What is ...?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A. ..."
  }}
]
"""

def generate_quiz(retriever, topic="", num_questions=5):
    query = topic if topic.strip() else "main topics and key concepts"
    results = retriever.retrieve(query, top_k=6)
    context = "\n\n".join([doc["document"] for doc in results]) if results else ""

    if not context:
        print("No context found for quiz generation")
        return []

    prompt = QUIZ_PROMPT.format(
        num_questions=num_questions,
        topic=topic if topic.strip() else "general concepts from the material",
        context=context[:6000]
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        quiz = json.loads(raw)
        print(f"Generated {len(quiz)} questions")
        return quiz
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return []