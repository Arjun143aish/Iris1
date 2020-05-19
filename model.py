import seaborn as sns
import pandas as pd
import numpy as np

iris = sns.load_dataset('iris')

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(iris,test_size = 0.3, random_state =123)

Train_X = Train.drop(['species'], axis =1)
Train_Y =Train['species'].copy()
Test_X = Test.drop(['species'], axis =1)
Test_Y = Test['species'].copy()

from sklearn.ensemble import RandomForestClassifier

M1_Model = RandomForestClassifier(random_state= 123).fit(Train_X,Train_Y)

Test_pred = M1_Model.predict(Test_X)

from sklearn.metrics import confusion_matrix,f1_score,recall_score, precision_score

Con_Mat = confusion_matrix(Test_pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

from sklearn.model_selection import GridSearchCV

n_tree = [50,75,100]
n_split = [100,200,300]
my_param_grid = {'n_estimators': n_tree,'min_samples_split': n_split}

Grid = GridSearchCV(RandomForestClassifier(random_state=123),param_grid= my_param_grid,
                    scoring= 'accuracy',cv = 5).fit(Train_X,Train_Y)

Grid_Df = pd.DataFrame.from_dict(Grid.cv_results_)

import pickle

pickle.dump(M1_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6.8,3,5.5,2.1]]))


                              