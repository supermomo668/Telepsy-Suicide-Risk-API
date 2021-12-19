import re
def lambda_handler(event, context):      
    # get Vocabs
    harm_json = '{"wound": 1, "obliterate": 1, "anguish": 1, "hurt": 1, "detriment": 1, "wounded": 1, "bolt down": 1, "injure": 1, "pop": 1, "pain": 1, "injury": 1, "wipe out": 1, "suffer": 1, "shoot down": 1, "damage": 1, "killing": 1, "suffering": 1, "weakened": 1, "ache": 1, "distress": 1, "harm": 1, "toss off": 1, "defeat": 1, "putting to death": 1, "stamp out": 1, "smart": 1, "kill": 1, "impairment": 1}'
    harm_vocab = json.loads(harm_json)
    search_words = '|'.join(list(harm_vocab.keys()))
    print("Search WORD:\n", search_words)
    # Search text
    for convo in event:
        for k, d in convo.items():
            if "content" in k: convo["content"] = convo.pop(k)
        text = convo["content"]
        if any(re.findall(search_words, text, re.IGNORECASE)):
            convo["risk"] = 1
            print("FOUND RISK")
        else:
            convo["risk"] = 0
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
if __name__ =="__main__":
    from pathlib import Path
    import os, json
    parent_path = Path(os.getcwd())
    #nltk.download('wordnet')
    with open(parent_path/"AlokAPITest.json") as jsonf:
        event_input = json.load(jsonf)
    #print(event_input)
    print(lambda_handler(event_input, context=0))