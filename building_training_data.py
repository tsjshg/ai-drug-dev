import pandas as pd
list_target = pd.read_csv('./list_of_known_targets.csv',sep=",")
embedding = pd.read_csv('./latent_space.txt',index_col=0,sep="\t")
embedding['OBJ'] = 'F'

d = {s: 'T' for s in list(list_target['Target gene name'])}
for i in embedding.index:
   if d.get(i) == None:
      embedding.at[i,'OBJ'] = 'F'   
   elif d.get(i) == 'T':
      embedding.at[i,'OBJ'] = d.get(i)
embedding.to_csv('training_data.txt',sep="\t")