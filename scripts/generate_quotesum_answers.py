import os
from together import Together
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time


# Load the QuoteSum dataset.
df = pd.read_json("/home/ubuntu/juraj/cloned/QuoteSum/v1/train.jsonl", lines=True)
values = df.values.tolist()
questions = list()
answers = list()
all_sources = list()

## Only every third entry in QuoteSum is unique.
for i in range(0, len(values), 3):
    val = values[i]
    sources = list()
    for j in range(6, len(val), 3):
        sources.append(val[j])
    
    q, a = val[2], val[3]
    questions.append(q)
    answers.append(a)
    all_sources.append(sources)
    
clean_answers = [a.replace("[" , "").replace("]", "") for a in answers]


all_messages = list()
for num_docs in [3]:

    all_messages = list()
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
            query = questions[idx]
            corpus = all_sources[idx]
            
            top_snippets = corpus[:num_docs]            
            top_context = ""
            for ts in top_snippets:
                top_context += ts
                top_context += " \n"

                
        
            messages =  [
                            {
                                "role": "user",
                                "content": f'''
                            Give a simple answer to the question based on the given context.

                        QUESTION:
                        ''' + questions[idx] + '''

                        CONTEXT:
                        ''' + top_context
                            }
                        ]

            all_messages.append(messages)



    os.environ["TOGETHER_API_KEY"] = "YOUR_API_KEY"

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    with open("mixtral_semqa_top"+str(num_docs)+"_gold.txt", "a") as f:
            for mess in all_messages:
                time.sleep(2)
                response = client.chat.completions.create(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    temperature=0,
                    max_tokens=400,
                    messages=mess,
                )

                resp = response.choices[0].message.content
                resp = resp.replace("\n", " ||| ")

                f.write(resp)
                f.write("\n")
