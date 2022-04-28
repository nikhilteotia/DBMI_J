from tkinter.tix import DirTree, Tree
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier


with open(r'C:\Users\Joy\Documents\GitHub\DBMI_J\models\clean_final.csv')as f:
    X_clean=pd.read_csv(f)
cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']
Array = X_clean.to_numpy()
X=Array[:,:19]
y = X_clean['target']
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)
with open(r'C:\Users\Joy\Documents\GitHub\DBMI_J\Final AdaBoostClassifier.joblib','rb')as f:
    adaboost=joblib.load(f)
with open(r'C:\Users\Joy\Documents\GitHub\DBMI_J\FinalBernoulliNB.joblib','rb')as f:
    bernaulli=joblib.load(f)
with open(r'C:\Users\Joy\Documents\Github\DBMI_J\FinalDecisionTreeClassifier.joblib','rb')as f:
    dtree=joblib.load(f)
with open(r'C:\Users\Joy\Documents\Github\DBMI_J\FinalGaussianMixture.joblib','rb')as f:
    gaussianmix=joblib.load(f)
with open(r'C:\Users\Joy\Documents\Github\DBMI_J\FinalGaussianNB.joblib','rb')as f:
    gaussian=joblib.load(f)
with open(r'C:\Users\Joy\Documents\GitHub\DBMI_J\FinalGradientBoosting.joblib','rb')as f:
    gradiant=joblib.load(f)
with open(r'C:\Users\Joy\Documents\GitHub\DBMI_J\FinalLogisticRegression.joblib','rb')as f:
    logistic=joblib.load(f)
with open(r'C:\Users\Joy\Documents\GitHub\DBMI_J\FinalLogisticRegressionCV.joblib','rb')as f:
    logisticcv=joblib.load(f)
with open(r'C:\Users\Joy\Documents\Github\DBMI_J\FinalRandomForestClassifier.joblib','rb')as f:
    forest=joblib.load(f)

    
clfs = {'GradientBoosting': gradiant,
        'LogisticRegression': logistic,
        'RandomForestClassifier': forest,
        'gaussianNB':gaussian,
        'adaboost' :adaboost,
        'GaussianMixture': gaussianmix,
        'DecisionTreeClassifier': dtree,
        'BernoulliNB': bernaulli,
        'LogisticRegressionCV':logisticcv

        }
cols = ['model', 'matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score', 'f1_score']
model_type='Non-Balanced'
models_report = pd.DataFrame(columns=cols)
conf_matrix = dict()
for clf, clf_name in zip(clfs.values(), clfs.keys()):
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    print('Accuracy:- ', accuracy_score(y_test, y_pred) * 100)
    print('f1 score:- ', f1_score(y_test, y_pred))

    tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score': metrics.roc_auc_score(y_test, y_score),
                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),
                         'precision_score': metrics.precision_score(y_test, y_pred),
                         'recall_score': metrics.recall_score(y_test, y_pred),
                         'f1_score': metrics.f1_score(y_test, y_pred)})

    models_report = models_report.append(tmp, ignore_index=True)
    conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=False)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate=False, pos_label=1)

    plt.figure(1, figsize=(6, 6))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')

    plt.title('ROC curve - {}'.format(model_type))
    plt.plot(fpr, tpr, label=clf_name)
    plt.legend(loc=2, prop={'size': 11})
    
plt.plot([0, 1], [0, 1], color='black')
plt.show()
print( models_report)

print(conf_matrix['LogisticRegression'])


print(conf_matrix['RandomForestClassifier'])

print(conf_matrix['GradientBoosting'])
print(conf_matrix['gaussianNB'])

print(conf_matrix['adaboost'])

print(conf_matrix['GaussianMixture'])
print(conf_matrix['DecisionTreeClassifier'])
print(conf_matrix['BernoulliNB'])
print(conf_matrix['LogisticRegressionCV'])