import bz2
import json
import os
import rich.progress
from guidance import models, select, assistant, user, system

import sentence_transformers
emb_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
import numpy as np

console = rich.get_console()

model = models.LlamaCpp("phi-2.Q5_K_M.gguf", echo=False)
#model = models.LlamaCpp("capybarahermes-2.5-mistral-7b.Q8_0.gguf", echo=False)
#model = models.OpenAI("gpt-3.5-turbo-1106", api_key=os.getenv("OPENAI_KEY"), echo=False)

with bz2.open('txts/1.json.bz2', 'rt') as f:
    data = json.load(f)
    page_data = data['features']['pages'][0]['body']

    token_counts_available = {}
    for t in page_data['tokenPosCount'].keys():
        for pos in page_data['tokenPosCount'][t].keys():
            token_counts_available[t] = token_counts_available.get(t, 0) + page_data['tokenPosCount'][t][pos]

    available_tokens = list(token_counts_available.keys())
    pred_txt = ""
    model += "TORCHLITE description: "
    while len(available_tokens) > 0:
        console.rule()
        console.print("[green]"+pred_txt+"[/green]")
        console.print("[blue]" + str(available_tokens) + "[/blue]")
        lm = model + select(available_tokens, name="token")
        model += lm['token'] + ' '
        pred_txt += lm['token'] + ' '
        token_counts_available[lm['token']] -= 1
        if token_counts_available[lm['token']] == 0:
            available_tokens.remove(lm['token'])
    console.print("[yellow]"+pred_txt+"[/yellow]")

    emb = emb_model.encode(pred_txt)
    with open('embeddings.csv', 'a') as f:
        f.write("\t".join(str(x) for x in emb) + "\n")
    with open('ids.csv', 'a') as f:
        f.write('llm\n')
