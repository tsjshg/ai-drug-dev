#import modules
import pandas as pd
import numpy as np
import sys
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score,roc_curve,auc,roc_auc_score,make_scorer,cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

np.random.seed(seed=113117)

dfile = "training_data.txt"
datada = pd.read_csv(dfile,sep="\t",index_col=0)

mytargetall = datada['OBJ']
df_TRUE = datada.query('OBJ == "T"')
df_FALSE = datada.query('OBJ == "F"')
df_TRUE['OBJ'] = 1
df_FALSE['OBJ'] = 0

PDATRUE = pd.DataFrame()
PDAFALSE = pd.DataFrame()
PDPRDA = pd.DataFrame()
PDSRDA = pd.DataFrame()

for i in range(100):   
   mytarget = ()
   dtrain_F = df_FALSE.sample(n=500)
   dtest_F = df_FALSE.drop(dtrain_F.index)
   dtrain = pd.concat([df_TRUE,dtrain_F],axis=0)
   dtest = dtest_F
   #print(dtest)

   mytarget = dtrain['OBJ']
   dtrain = dtrain.drop('OBJ',axis=1)
   
   mytargettest = dtest['OBJ']
   dtest = dtest.drop('OBJ',axis=1)
 
   #metrics to evaluate models
   kappa_scorer = make_scorer(cohen_kappa_score)
   roc_auc = make_scorer(roc_auc_score)
   
   #pipe line
   pipe = imbPipeline([
          ('smote', SMOTE(sampling_strategy=1.0,k_neighbors=2,random_state=117117)),
          ('clf', XGBClassifier(seed=0))
          ])
   
   #tuning parameter
   param_grid = {
                 "clf__max_depth" : [1,2,3,5,10],
                 "clf__booster" : ['gblinear'],
                 "clf__n_estimators" : [100],
                 "clf__learning_rate" : [0.01,0.1,0.5],
                 "clf__gamma" : [0,0.3],
                 "clf__objective" : ['binary:logistic'],
                 "clf__reg_lambda" : [0,0.1,1.0],
                 "clf__reg_alpha" : [0,0.1,1.0]
                 }
   
   #grid search setting
   grid = GridSearchCV(pipe, param_grid=param_grid, cv = 5, scoring=roc_auc, n_jobs=30, refit=True, verbose=3)
   
   #model optimization
   grid.fit(dtrain.values,mytarget)
   
   PDA = pd.DataFrame(grid.predict_proba(dtest.values),index=dtest.index)
   PDATRUE = pd.concat([PDATRUE,PDA[1]],axis=1)
   PDAFALSE = pd.concat([PDAFALSE,PDA[0]],axis=1)

   PDS = pd.concat([pd.Series(grid.cv_results_['mean_test_score'][grid.best_index_]),pd.Series(grid.cv_results_['std_test_score'][grid.best_index_])],axis=1)
   PDSRDA = pd.concat([PDSRDA,PDS],axis=0)
   PDPRDA = pd.concat([PDPRDA,pd.Series(grid.cv_results_['params'][grid.best_index_])],axis=1)

PDADA = pd.concat([PDATRUE.mean(axis=1),PDAFALSE.mean(axis=1),mytargetall],axis=1)
PDADA = PDADA.rename(columns={0:'probability for drug target', 1:'probability for non-drug target', 'OBJ':'answer'})
PDADA.to_csv('Prediction_results_for_putative_targets.txt',index=True,sep="\t")
PDSRDA.to_csv('Scores_for_100_iterations.txt',index=True,sep="\t")
PDPRDA.to_csv('Best_parameters_for_100_iterations.txt',index=True,sep="\t")

