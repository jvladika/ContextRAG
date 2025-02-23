import os
from together import Together
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import json

## Load BioASQ data
with open('training11b.json') as f:
    d = json.load(f)

all_data = d["questions"]
counter = 0
sum_entries = list()

# Consider only summary type questions
for entry in all_data:
    if entry["type"]=="summary":
        counter += 1
        sum_entries.append(entry)

all_snippets = list()
all_questions = list()
all_ideals = list()
all_documents = list()

for entry in sum_entries:
    all_questions.append(entry["body"])
    all_snippets.append(entry["snippets"])
    all_ideals.append(entry["ideal_answer"])
    all_documents.append(entry["documents"])


# Load the sentence transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get all snippets
all_snips = list()
for asnip in all_snippets:
    docsnips = list()
    for snip in asnip:
        docsnips.append(snip["text"])

    all_snips.append(docsnips)

# Get the top k most similar snippets, where k € {0, 1, 3, 5, 10}
# Generate an answer based on the top k snippets
for num_docs in [0, 1, 3, 5, 10]:

    all_messages = list()
    questions = all_questions
    for idx in range(len(questions)):
    
        if num_docs == 0:
            messages =  [
                            {
                                "role": "user",
                                "content": f'''
                            Give a simple answer to the question based on your best knowledge.

                        QUESTION:
                        ''' + questions[idx] 
                            }
                        ]
            all_messages.append(messages)

        else:
            query = all_questions[idx]
            corpus = all_snips[idx]

            corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
            query_embedding = embedder.encode(query, convert_to_tensor=True)

            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

            topk = min(len(corpus), num_docs)
            top_results = torch.topk(cos_scores, k=topk)

            top_index = top_results.indices[0].item()
            top_snippets = [corpus[idx] for idx in top_results.indices]
            top_context = ""
            for ts in top_snippets:
                top_context += ts
                top_context += " \n"
        
            messages =  [
                            {
                                "role": "user",
                                "content": f'''
                            Give a simple answer to the question based on the context.

                        QUESTION:
                        ''' + questions[idx] + '''

                        CONTEXT:
                        ''' + top_context
                            }
                        ]

            all_messages.append(messages)

    os.environ["TOGETHER_API_KEY"] = "YOUR_API_KEY"

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


    # Change file name to your own
    with open("qwen-7b_answers_top"+str(num_docs)+"_gold.txt", "a") as f:
        for mess in all_messages:

            ## CHANGE THE MODEL NAME TO YOUR OWN
            response = client.chat.completions.create(
                model="Qwen/Qwen1.5-7B-Chat",
                temperature=0,
                max_tokens=500,
                messages=mess,
            )

            resp = response.choices[0].message.content
            resp = resp.replace("\n", " ||| ")

            f.write(resp)
            f.write("\n")
