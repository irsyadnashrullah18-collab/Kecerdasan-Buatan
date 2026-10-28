from transformers import pipeline

question_answerer = pipeline('question-answering')

# Simpan ke variabel lalu cetak dengan print()
result = question_answerer(
    question="What is my name?",
    context="My name is Clara and I live in Berkeley."
)

print(result)