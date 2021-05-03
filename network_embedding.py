
import argparse

parser = argparse.ArgumentParser(description='Network Embedding')
parser.add_argument('--pin', default='PIN_data.csv', help='Protein interaction data')
parser.add_argument('--model', default='network_embedding_model.hdf5', help='Pre trained model')
parser.add_argument('--output', default='latent_space.txt', help='Output file name')

args = parser.parse_args()

import numpy as np
import pandas
from pandas import DataFrame
from pandas import read_csv
from keras.models import load_model


PIN_DATA = args.pin
MODEL = args.model
OUTPUT = args.output

x_test = read_csv(PIN_DATA)
y_test = x_test['GENE']
x_test = x_test.drop(['GENE'], axis=1)
x_test = x_test.values

encoder = load_model(MODEL) 
latent_space = np.array(encoder.predict(x_test))
latent_space = pandas.DataFrame(latent_space)
latent_space.index = y_test
latent_space.to_csv(OUTPUT, sep='\t')
