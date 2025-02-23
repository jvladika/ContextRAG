import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

sentence_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO").cuda()
abstracts = pd.read_csv("/mnt/mydisk/PubMed/pubmed_landscape_abstracts.csv")

##GENERATE EMBEDDINGS
for step in range(200, 100, -1):
    abstracts_slice = abstracts[step*100000:(step+1)*100000].AbstractText.tolist()
    encoded_slice = sentence_model.encode(abstracts_slice)

    with open("../PubMed/embeddings/" + str(step) + ".npy",'wb') as f:
        np.save(f, encoded_slice)
        
          
