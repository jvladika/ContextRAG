import wikipediaapi
from nltk.tokenize import sent_tokenize
import pandas as pd 
import torch
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval

import pandas as pd
import numpy as np

df = pd.read_json("/home/ubuntu/juraj/cloned/QuoteSum/v1/train.jsonl", lines=True)

values = df.values.tolist()
questions = list()
answers = list()
all_sources = list()

for i in range(0, len(values), 3):
    val = values[i]
    sources = list()
    for j in range(6, len(val), 3):
        sources.append(val[j])
    
    q, a = val[2], val[3]
    questions.append(q)
    answers.append(a)
    all_sources.append(sources)
    

wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
with open("../wikititles.txt", "r") as f:
    lines = f.readlines()


titles = list()
for l in lines:
    titles.append(literal_eval(l))
    
    
claim_sentences = list()
claim_idx = 0
for res in titles:
    if claim_idx%50==0:
        print(claim_idx)
    all_sentences = list()
    for article in res:
        name = article
        page = wiki_wiki.page(name)
        try:
            full_text = page.text
        except:
            continue   
                    
        sentences = sent_tokenize(full_text)
        all_sentences.extend(sentences)
        #all_sentences = [s.lower() for s in all_sentences]
    
    claim_sentences.append(all_sentences)
    claim_idx += 1
        
print("collected all sentences from abstracts!")   

import pickle

with open('claim_sentences.pkl', 'wb') as file:
    pickle.dump(claim_sentences, file)

model = SentenceTransformer('model')
model.to("cuda")
print("loaded sentence model!")


top_sentences = list()
for idx in range(len(questions)):
    claim = questions[idx]
    print(idx)
    sents = claim_sentences[idx]
    
    sents_embeddings = model.encode(sents, convert_to_tensor=True).to("cuda")
    claim_embedding = model.encode(claim, convert_to_tensor=True).to("cuda")
    cos_scores = util.cos_sim(claim_embedding, sents_embeddings)[0]
    
    k_value = min(len(cos_scores), 30)
    top_results = torch.topk(cos_scores, k=k_value)
    
    #for score, i in zip(top_results[0], top_results[1]):
    #    print(sents[i], "(Score: {:.4f})".format(score))
    np_results = top_results[1].detach().cpu().numpy()
    top_sentences.append(np_results)
print("done with cosine similarity calculation")


selected_sentences = list()
for idx in range(len(questions)):
    top = top_sentences[idx]
    top = np.sort(top)
    sents = np.array(claim_sentences[idx])[top]    
    selected_sentences.append(sents)
    
joint_list = list()
for idx in range(len(questions)):
    joint = questions[idx] + " [SEP] "
    for s in selected_sentences[idx]:
        joint += s
        joint += " ||| "
    joint_list.append(joint)

with open("bm25wiki_joint_semqa.txt", "w") as f:
	for example in joint_list:
		f.write(example.replace("\n", " "))
		f.write("\n")
