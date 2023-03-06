import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

movieId = np.arange(1,31)

genome_scores_dataframe = pd.DataFrame(columns=['movieId', 'tagId', 'relevance'])
print("Ciao")
print(genome_scores_dataframe)

tag_id = []
for i in range(1, 1129):
    tag_id.append(i)

cnt = 0
relevance = []
for j in range(1, 31):
    for i in range(1, 1129):
        data = {'movieId': int(j), 'tagId': tag_id[int(i) - 1], 'relevance': np.random.random() / 100}
        genome_scores_dataframe = genome_scores_dataframe.append(data, ignore_index=True)


genome_scores_dataframe.to_csv('./genome-scores.csv', index = False)