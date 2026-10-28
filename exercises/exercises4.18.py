# Exercise 4.18: Ubah Example 4.22 dengan teks yang berbeda, lalu komentari hasilnya.

import nltk

for res in ['punkt', 'punkt_tab',
            'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
            'maxent_ne_chunker', 'maxent_ne_chunker_tab',
            'words']:
    nltk.download(res)

sentence = """
Sundar Pichai is the chief executive of Google, which is based in Mountain View,
California. Satya Nadella leads Microsoft in Redmond. In 2012, researchers at the
University of Toronto built AlexNet, and Geoffrey Hinton became famous in Canada
and around the world. Jakarta is the capital of Indonesia.
"""
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)

# Ringkasan: hanya named entity yang ditemukan
print("\nNamed entities:")
for subtree in entities:
    if hasattr(subtree, 'label'):
        print(f"  {subtree.label():<13} {' '.join(word for word, tag in subtree.leaves())}")

