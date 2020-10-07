import numpy as np
import pandas
from pandas import DataFrame
from pandas import read_csv
from keras.models import load_model

x_test = read_csv("PIN_data.csv")
y_test = x_test['GENE']
x_test = x_test.drop(['GENE'],axis=1)
x_test = x_test.values

encoder = load_model('./network_embedding_model.hdf5')
latent_space = np.array(encoder.predict(x_test))
latent_space = pandas.DataFrame(latent_space)
latent_space.index = y_test
latent_space.to_csv('latent_space.txt',sep='\t')
