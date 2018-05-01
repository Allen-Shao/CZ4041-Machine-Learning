
# coding: utf-8

# In[2]:

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# In[3]:

train_data = pd.DataFrame.from_csv("./data/train.csv")
test_feature = pd.DataFrame.from_csv("./data/test.csv")


# In[4]:

train_labels = train_data['species'].values
train_feature = train_data[[i for i in train_data.columns if i != 'species']].values


# In[5]:

minmax_scaler = MinMaxScaler()
train_feature = minmax_scaler.fit_transform(np.array(train_feature))
test_feature = minmax_scaler.transform(np.array(test_feature))


# In[6]:

def train_validation_split(x, y):
    idx = (list(range(len(y))))
    np.random.shuffle(idx)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(x) != np.ndarray:
        x = np.array(x)
    y = y[idx]
    x = x[idx]
    
    vali_idx = []
    s = {}
    for k in set(y):
        s[k] = 2
        
    for i in range(len(y)):
        if s[y[i]] > 0:
            vali_idx.append(i)
            s[y[i]] = s[y[i]] - 1
            
    train_idx = [i for i in range(len(y)) if i not in vali_idx]
    return  x[train_idx], x[vali_idx], y[train_idx], y[vali_idx]

X_train, X_vali, y_train, y_vali = train_validation_split(train_feature, train_labels)


# In[9]:



classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)


for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_vali)
    acc = accuracy_score(y_vali, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_vali)
    ll = log_loss(y_vali, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[10]:

def prepare_submission_file(clf, file_name):
    if (os.path.exists('./baseline_submission/') is False):
        os.mkdir('./baseline_submission/')
    res = clf.predict_proba(test_feature)
    max_idx = np.argmax(res,axis=1)
    zeros = np.zeros(shape = res.shape)
    for i in range(len(max_idx)):
        zeros[i][max_idx[i]] = 1
    submission = pd.DataFrame(zeros, columns=clf.classes_)
    submission.insert(0, 'id', pd.DataFrame.from_csv("./data/test.csv").index.values)
    submission.to_csv("./baseline_submission/" + file_name + ".csv", index=False)


# In[11]:

# best_clf = classifiers[2]
# log.to_csv("./baseline_submission/log.csv")
# for i in classifiers:
i = KNeighborsClassifier(3)
i.fit(train_feature, train_labels)
prepare_submission_file(i,i.__class__.__name__)


# In[ ]:



