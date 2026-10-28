import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline

# Memuat model
question_answerer = pipeline(
    'question-answering',
    model='distilbert/distilbert-base-cased-distilled-squad'
)

context = (
    "Surabaya is the capital of East Java province in Indonesia. "
    "It is the second largest city in the country after Jakarta, and it is "
    "located on the northeastern coast of Java island. The city is known as "
    "the City of Heroes because of the Battle of Surabaya in November 1945."
)

questions = [
    "What is the capital of East Java?",
    "Which city is larger than Surabaya?",
    "Where is Surabaya located?",
    "Why is Surabaya called the City of Heroes?",
    "When did the Battle of Surabaya happen?",
    "Who is the mayor of Surabaya?",  # Jawabannya TIDAK ada di teks
]

for q in questions:
    result = question_answerer(question=q, context=context)
    print(f"Q: {q}\n   A: {result['answer']!r}  (score={result['score']:.3f})\n")