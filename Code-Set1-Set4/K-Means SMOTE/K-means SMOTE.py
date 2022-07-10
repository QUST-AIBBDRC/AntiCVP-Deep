import numpy as np
import pandas as pd 
# import imbalanced_learn as imblearn
# from imblearn.ensemble import EasyEnsemble
from sklearn.preprocessing import scale,StandardScaler 
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import NeighbourhoodCleaningRule
#from imblearn.ensemble import BalanceCascade,EasyEnsemble 
from imblearn.combine import SMOTEENN,SMOTETomek 
from imblearn.over_sampling import ADASYN, RandomOverSampler,SMOTE 
#from imblearn.ensemble import balance_cascade, easy_ensemble
# from imblearn.ensemble import BalanceCascade,EasyEnsemble 
from sklearn.preprocessing import scale,StandardScaler 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
#print("jjfj")
data_=pd.read_csv(r'Training_Set1_fusion.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((95,1))#Value can be changed   int(m1/2)
label2=np.zeros((1399,1))
#label1=np.ones((int(m1/2),1))#Value can be changed
#label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)

X=shu
y=label#.astype('int64')



sm = KMeansSMOTE(cluster=0.0001,random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)


shu=X_resampled
X1=scale(shu)
y1=y_resampled

#shu2 =X_resampled
#shu3 =y_resampled
data_csv = pd.DataFrame(data=X1)
data_csv.to_csv('KMeansSMOTE_Fusion_Training_Set1.csv')
data_csv = pd.DataFrame(data=y1)
data_csv.to_csv('label_KMeansSMOTE_Fusion_Training_Set1.csv')