import bz2
import json
import os
import rich.progress
from guidance import models, select, assistant, user, system
import itertools

import sentence_transformers
emb_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
import numpy as np

console = rich.get_console()

#model = models.LlamaCpp("Mixtral-8x7B-v0.1.Q2_K.gguf", echo=False)
model = models.LlamaCpp("Phi-3-mini-4k-instruct-fp16.gguf", echo=False)
#model = models.LlamaCpp("capybarahermes-2.5-mistral-7b.Q8_0.gguf", echo=False)
#model = models.OpenAI("gpt-3.5-turbo-1106", api_key=os.getenv("OPENAI_KEY"), echo=False)

with bz2.open('txts/f.json.bz2', 'rt') as f:
    data = json.load(f)['data']
    page_data = data['features']['pages'][24]['body']

    token_counts_available = {}
    for t in page_data['tokenPosCount'].keys():
        for pos in page_data['tokenPosCount'][t].keys():
            token_counts_available[t] = token_counts_available.get(t, 0) + page_data['tokenPosCount'][t][pos]

    available_tokens = list(token_counts_available.keys())
    pred_txt = ""
    model += "<|user|>\nStart of Frankenstein:<|end|>\n<|assistant|>\n"
    while len(available_tokens) > 0:
        avail_tokens = list(itertools.chain.from_iterable([[k]*v for k,v in token_counts_available.items()]))
        tok_groups = list(map(lambda ts: " ".join(ts), itertools.permutations(avail_tokens, 1)))
        console.rule()
        console.print("[green]"+pred_txt+"[/green]")
        console.print("[blue]" + str(available_tokens) + "[/blue]")
        lm = model + select(tok_groups, name="token")
        model += lm['token'] + ' '
        pred_txt += lm['token'] + ' '
        ts = lm['token'].split()
        for t in ts:
            print(t)
            token_counts_available[t] -= 1
            if token_counts_available[t] == 0:
                available_tokens.remove(t)
    console.print("[yellow]"+pred_txt+"[/yellow]")

    emb = emb_model.encode(pred_txt)
    with open('embeddings.csv', 'a') as f:
        f.write("\t".join(str(x) for x in emb) + "\n")
    with open('ids.csv', 'a') as f:
        f.write('llm\n')
