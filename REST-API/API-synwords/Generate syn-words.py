# Generate 
import nltk, json
from nltk.corpus import wordnet
def get_synonyms(bag_of_words, lim_lemma=3):
    # reduce lemma to reduce search time
    synonyms = []
    for keyword in bag_of_words:
        for syn in wordnet.synsets(keyword):
            syn_list = syn.lemmas()
            if lim_lemma > 0:
                syn_list = syn_list[:lim_lemmas]
            for l in syn_list:
                word = l.name()
                if "_" in word:
                    word = " ".join(word.split("_"))
                synonyms.append(word)
    return set(synonyms)

if __name__ =="__main__":
    from pathlib import Path
    import os
    parent_path = Path(os.getcwd())
    #nltk.download('wordnet')
    harm_words = ["harm", "kill", "hurt", "injure","self-harm","cut", "murder",\
                "terminate","end","life"]
    synonyms = get_synonyms(harm_words, lim_lemma=0)
    syn = dict.fromkeys(synonyms, 1)
    json_out = json.dumps(syn)
    fp = "synonym_words.json"
    with open(parent_path/fp,"w") as f:
        json.dump(json_out, f)
    #print("Json dump:",json_out)
    #print("Json load:",json.loads(json_out))