
import pandas as pd

import  argparse
parser = argparse.ArgumentParser(description='Building training data')
parser.add_argument('--targets', default='list_of_known_targets.csv', help='Known target list')
parser.add_argument('--emb', default='latent_space.txt', help='The output file of network_embedding.py')
parser.add_argument('--output', default='training_data.txt', help='Output file name for Xggoost_training.py')

args = parser.parse_args()

list_target = pd.read_csv(args.targets, sep=",")
embedding = pd.read_csv(args.emb, index_col=0, sep="\t")
embedding['OBJ'] = 'F'

d = {s: 'T' for s in list(list_target['Target gene name'])}
for i in embedding.index:
   if d.get(i) == None:
      embedding.at[i,'OBJ'] = 'F'   
   elif d.get(i) == 'T':
      embedding.at[i,'OBJ'] = d.get(i)

embedding.to_csv(args.output, sep="\t")