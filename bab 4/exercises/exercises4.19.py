# Exercise 4.19: Ubah Example 4.23 dengan beberapa kalimat berbeda (sebagian positif,

from transformers import pipeline

classifier = pipeline('sentiment-analysis',
                      model='distilbert-base-uncased-finetuned-sst-2-english')

sentences = [
    "This is a good movie.",
    "I love this phone, the battery lasts for days.",
    "The food was delicious and the staff were very friendly.",
    "This is the worst movie I have ever seen.",
    "The service was slow and the room was dirty.",
    "I am very disappointed with this product.",
    "The movie was not bad.",                       
    "It is a movie about a man and his dog.",       # netral
]

for sent in sentences:
    result = classifier(sent)[0]
    print(f"{result['label']:<9} {result['score']:.4f}  | {sent}")

