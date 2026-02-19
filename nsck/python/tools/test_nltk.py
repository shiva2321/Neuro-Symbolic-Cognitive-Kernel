
import nltk

print("Downloading 'averaged_perceptron_tagger_eng'...")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')

text = "The quick brown fox jumps over the lazy dog."
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)

print(f"Tagged: {tags}")

# Verify mapping
mapped = []
for word, tag in tags:
    simple_tag = "UNK"
    if tag.startswith("NN"): simple_tag = "NN"
    elif tag.startswith("VB"): simple_tag = "VB"
    elif tag.startswith("DT"): simple_tag = "DT"
    elif tag.startswith("JJ"): simple_tag = "JJ" # Adjective
    mapped.append((word, simple_tag))
    
print(f"Mapped: {mapped}")
