# Example 4.23 Transformers: sentiment analysis
# https://github.com/huggingface/transformers
# Library: transformers dan torch   ->  pip install transformers torch
#  https://aka.ms/vs/16/release/vc_redist.x64.exe)
from transformers import pipeline

classifier = pipeline('sentiment-analysis',
                      model='distilbert-base-uncased-finetuned-sst-2-english')
result = classifier('This is a good movie.')
print(result)   # [{'label': 'POSITIVE', 'score': 0.99986...}]
