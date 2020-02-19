# Chunr prevention 
target customers who will churn based on 3 months sales

-*- coding: utf-8 -*-

"""

Created on Mon Nov 13 17:48:53 2017

 

@author: j-ago

"""

#%%必要な全てのデータを格納する

 

# data processing

# package for high-performance, easy-to-use data structures and data analysis

import pandas as pd

 

# linear algebra

# fundamental package for scientific computing

import numpy as np

 

# data visualization

# for making plots with seaborn

import seaborn as sns

sns.set_style('whitegrid')

 

# plotting

import matplotlib.pyplot as plt

# making plots inside notebook

%matplotlib inline

 

# sklearn

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import RFE

from sklearn.feature_selection import f_classif

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.calibration import CalibratedClassifierCV

 

# scipy for feature normalization

from scipy import stats

 

import pandas as pd

 

from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

 

#%%トレーニングデータを格納する y=目的変数とする

 

df= pd.read_csv('D:/CHURN_TRAINNING.csv',index_col =0)

y=df.RESPONSE

#%%

 

df.dtypes

#%%バイナリのデータのみList 1格納する　⇒　ここから標準化　実数と割合を標準化

 

df_max = df.max().reset_index()

df_reordered = df_max['index'].str.contains('Under40|Age|Over80|flg').rename('flg')

df_max2 = pd.concat([df_max,df_reordered], axis = 1)

list_1_original = df_max2[df_max2.flg == True]

list_1 = list_1_original['index'].tolist()

#%%実数のカラムのみList 2に格納する

df_max = df.max().reset_index()

df_reordered = df_max['index'].str.contains('sales|pen|cnt|share|s_|age|recency|YEAR1|ITVL').rename('flg')

df_max2 = pd.concat([df_max,df_reordered], axis = 1)

list_2_original = df_max2[df_max2.flg == True]

list_2 = list_2_original['index'].tolist()

 

 

 

#%%もとデータからlist 1を抜く、Ｌｉｓｔ２を抜く　list1 list2　をX1 X2に分ける　それをdata2にする

x_1 = df.drop(list_1,axis = 1 )

x_2 = df.drop(list_2,axis = 1 )

 

# それぞれのデータから目的変数を抜く

x_1 = x_1.drop('RESPONSE' ,axis = 1 )

x_2 = x_2.drop('RESPONSE' ,axis = 1 )

 

#データを標準化し、data_2にデータを結合する

# standardization

#data_1 = (x_1 - x_1.mean()) / (x_1.std()).fillna(0) 

 

data_1_scaled = StandardScaler().fit_transform(x_1)

 

data_1 = pd.DataFrame(data_1_scaled, columns=x_1.columns,index=x_1.index)

 

# concatenate features

data_2 = pd.concat([y,data_1,x_2],axis=1)

 

 

#%%dデータを確認する

data_2.columns

data_2.describe()

#%%d

sns.distplot(data_2['total_sales'],

             kde=True,

             bins=50)

plt.show()

#%%ｙを目的変数、Ｘを説明変数に設定する

# y includes our labels and x includes our features

y = data_2.RESPONSE

list = ['RESPONSE']

X = data_2.drop(list,axis = 1 )

 

 

#%%Gradient Boosting Classifier を格納し、トレーニンググループとテストグループに分割する

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

 

#%%予測モデルの比較

 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

 

 

 

# prepare configuration for cross validation test harness

seed = 7

 

# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('GDBR',GradientBoostingClassifier()))

models.append(('RDFC',RandomForestClassifier()))

 

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

              kfold = model_selection.KFold(n_splits=5, random_state=seed)

              cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

              results.append(cv_results)

              names.append(name)

              msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

              print(msg)

   

    # boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()

 

 

#%%予測モデルをイニシャライズし、チューニングを変更する。トレーニングデータにモデルを当てはめ、スコアを確認する

 

 

gbrt=GradientBoostingClassifier(random_state=42,learning_rate = 0.03,max_depth = 5)

 

gbrt.fit(X_train,y_train)

print("Accuracy on train set:{:.3f}".format(gbrt.score(X_train,y_train)))

 

 

#%%テストデータに当てはめてテストデータのスコアを確認する

 

print("Accuracy on test set:{:.3f}".format(gbrt.score(X_test,y_test)))

 

 

#%%confusion matrixで精度を確認する

pred_gbrt = gbrt.predict(X_test)

 

from sklearn.metrics import confusion_matrix

 

confusion = confusion_matrix(y_test,pred_gbrt)

print("Confusion matrix:\n{}".format(confusion))

#%%classification reportで精度を確認する

target_names =['no_churn','churn']

print(classification_report(y_test,pred_gbrt,target_names=target_names))

#%%ROC scoreで精度を確認する

print(roc_auc_score(y_test,pred_gbrt))

 

#%%特徴量を算出し、データに吐き出す

feature = pd.DataFrame(gbrt.feature_importances_).rename(columns = {0:'score'})

columns = pd.DataFrame(X_train.columns.values).rename(columns = {0:'feature_importance'})

data = pd.concat([columns,feature],axis ='columns')

data = data[data['score'] > 0].sort_values('score',ascending=False)

data.to_csv("D:/feature_importance.csv")

 

#%%CHECK CUSTOMER FEATURE IMPORTANCE BY RESPONSE

grouped=df.groupby(['RESPONSE'])

 

grouped =grouped.agg({

       'RESPONSE':  'count',

       'order_cnt': ['mean','median','std'],

       'brand_cnt': ['mean','median','std'],

       'ITVL_MIN_ALL': ['mean','median','std'],

        'ITVL_LATEST': ['mean','median','std'],

         'ITVL_MAX_ALL': ['mean','median','std'],

       'ITVL_MEAN_ALL': ['mean','median','std'],

       'skn_cnt': ['mean','median','std'],

       'cat_cnt': ['mean','median','std'],

       'age': ['mean','median','std'],

       'cat_2_cnt': ['mean','median','std']

      

       }).round(1).sort_index(ascending=False)

 

grouped.to_excel(excel_writer = "D:/feature_importance_info.xlsx",

                      sheet_name="segment",

                          startrow=1,

                          startcol=1)

 

#%%CHECK CUSTOMER FEATURE IMPORTANCE BY RESPONSE

scores = gbrt.predict_proba(X_test)

 

preds = [s[1] > 0.5 for s in scores]

 

preds_default = gbrt.predict(X_test)

 

all(preds == preds_default)

 

#%%CHECK CUSTOMER FEATURE IMPORTANCE BY RESPONSE

 

t_range=[0.3,0.35,0.4,0.45,0.5,0.75]

 

preds = [[s[1] > thr for s in scores] for thr in t_range]

 

accuracies = [accuracy_score(y_test, p) for p in preds]

 

f1_scores = [f1_score(y_test, p) for p in preds]

 

recall_score = [recall_score(y_test, p) for p in preds]

 

print (t_range[np.argmax(accuracies)], t_range[np.argmax(f1_scores)])

#%%

print(t_range[np.argmax(recall_score)])

 

#%%cost fp

 

 

def my_scorer(y_test,y_est, cost_fp= 34596, cost_fn = 1000):

    tn, fp, fn, tp =confusion_matrix(y_test,pred_gbrt).ravel()

    return cost_fp * fp + cost_fn * fn

 

gbrt = gbrt.fit(X_train, y_train)

print(my_scorer(y_test, gbrt.predict(X_test)))

 

#%%cost tp

 

 

def my_scorer(y_test,y_est, cost_tp= 34596, cost_fn = 1000):

    tn, fp, fn, tp =confusion_matrix(y_test,pred_gbrt).ravel()

    return cost_tp * tp

 

gbrt = gbrt.fit(X_train, y_train)

print(my_scorer(y_test, gbrt.predict(X_test)))

 

 

#%%validationデータを格納する y=目的変数とする

 

df= pd.read_csv('D:/CHURN_VALIDATION.csv',index_col =0)

y=df.RESPONSE

#%%バイナリのデータのみList 1格納する　⇒　ここから標準化　実数と割合を標準化

 

df_max = df.max().reset_index()

df_reordered = df_max['index'].str.contains('Under40|Age|Over80|flg').rename('flg')

df_max2 = pd.concat([df_max,df_reordered], axis = 1)

list_1_original = df_max2[df_max2.flg == True]

list_1 = list_1_original['index'].tolist()

#%%実数のカラムのみList 2に格納する

df_max = df.max().reset_index()

df_reordered = df_max['index'].str.contains('sales|pen|cnt|share|s_|age|recency|YEAR1|ITVL').rename('flg')

df_max2 = pd.concat([df_max,df_reordered], axis = 1)

list_2_original = df_max2[df_max2.flg == True]

list_2 = list_2_original['index'].tolist()

 

 

#%%もとデータからlist 1を抜く、Ｌｉｓｔ２を抜く　list1 list2　をX1 X2に分ける　それをdata2にする

x_1 = df.drop(list_1,axis = 1 )

x_2 = df.drop(list_2,axis = 1 )

 

# それぞれのデータから目的変数を抜く

x_1 = x_1.drop('RESPONSE' ,axis = 1 )

x_2 = x_2.drop('RESPONSE' ,axis = 1 )

 

#データを標準化し、data_2にデータを結合する

# standardization

#data_1 = (x_1 - x_1.mean()) / (x_1.std()).fillna(0) 

 

data_1_scaled = StandardScaler().fit_transform(x_1)

 

data_1 = pd.DataFrame(data_1_scaled, columns=x_1.columns,index=x_1.index)

 

# concatenate features

data_2 = pd.concat([y,data_1,x_2],axis=1)

 

#%%Gradient Boosting Classifier を格納し、トレーニンググループとテストグループに分割する

 

X_validation = data_2.drop ('RESPONSE' ,axis = 1 )

y_validation = data_2.RESPONSE

 

#%%validationデータに当てはめてテストデータのスコアを確認する

 

print("Accuracy on test set:{:.3f}".format(gbrt.score(X_validation,y_validation)))

#%%confusion matrixで精度を確認する

pred_gbrt = gbrt.predict(X_validation)

 

from sklearn.metrics import confusion_matrix

 

confusion = confusion_matrix(y_validation,pred_gbrt)

print("Confusion matrix:\n{}".format(confusion))

#%%VALIDATION_HOLDOUTで全顧客のPROBAを算出する

 

# Create empty list: dfs

customer = []

score = []

 

# Loop over 'WDI.csv'

for chunk in pd.read_csv('D:/CHURN_HOLDOUT_ALL.csv',index_col = 0, chunksize=10000):

   

    df_flag = chunk.max().reset_index()

    df_reordered = df_flag['index'].str.contains('Under40|Age|Over80|flg').rename('flg')

    df_flag2 = pd.concat([df_flag,df_reordered], axis = 1)

    list_1_original = df_flag2[df_flag2.flg == True]

    list_1 = list_1_original['index'].tolist()

 

    df_num = chunk.max().reset_index()

    df_reordered2 = df_num['index'].str.contains('sales|pen|cnt|share|s_|age|recency|YEAR1|ITVL').rename('flg')

    df_num2 = pd.concat([df_num,df_reordered2], axis = 1)

    list_2_original = df_num2[df_num2.flg == True]

    list_2 = list_2_original['index'].tolist()

 

 

    x_1 = chunk.drop(list_1,axis = 1 ).fillna(0)

    x_2 = chunk.drop(list_2,axis = 1 ).fillna(0)

 

    # それぞれのデータから目的変数を抜く

    x_1 = x_1.drop('RESPONSE' ,axis = 1 )

    x_2 = x_2.drop('RESPONSE' ,axis = 1 )

 

    # standardization

    data_1_scaled = StandardScaler().fit_transform(x_1)

    

    data_1 = pd.DataFrame(data_1_scaled, columns=x_1.columns,index=x_1.index)

 

    # concatenate features

    data_2 = pd.concat([data_1,x_2],axis=1)

   

    # Create the first Series

    result = gbrt.predict_proba(data_2)

    result = pd.DataFrame(result)

    result = result.iloc[:,1]

   

 

    # Create the second Series

    customer_id = chunk.loc[:,['app_sales','app_share','ec_sales','ec_pen','tsr_pen','ivr_pen']]

 

   

 

    # Append the filtered chunk to the list dfs

    customer.append(customer_id)

    score.append(result)

   

#%%to_csv

customer_pd=pd.concat(customer).reset_index()

 

score_pd=pd.concat(score)

score_pd=pd.DataFrame(score_pd).reset_index(drop=True)

 

merge = customer_pd.join(score_pd, how='inner')

merge = merge.rename(columns={1: 'proba'})

merge.to_csv("D:/customer.csv")   

 

#%%validationのデータを格納する

 

df= pd.read_csv('D:/validation_holdout.csv',index_col =0)

df = df.reset_index()

 

#%%to_csv　オリジナルのバリデーションデータをマージ


merge2 = merge.merge(df, how = 'inner', on = 'CUSTOMER_ID')

#%%

merge2.iloc[0]

 

#%%予測適合確率ごとに顧客をセグメントする

 

def pivot(merge):

    if ((merge['proba'] >= 0) and (merge['proba']  <= 0.1)):

        return '0-0.1'

    elif ((merge['proba'] > 0.1) and (merge['proba']  <= 0.2)):

        return '0.1-0.2'

    elif ((merge['proba'] > 0.2) and (merge['proba']  <= 0.3)):

        return '0.2-0.3'

    elif ((merge['proba'] > 0.3) and (merge['proba']  <= 0.4)):

        return '0.3-0.4'

    elif ((merge['proba'] > 0.4) and (merge['proba']  <= 0.5)):

        return '0.4-0.5'

    elif ((merge['proba'] > 0.5) and (merge['proba']  <= 0.6)):

        return '0.5-0.6'

    elif ((merge['proba'] > 0.6) and (merge['proba']  <= 0.7)):

        return '0.6-0.7'

    elif ((merge['proba'] > 0.7) and (merge['proba']  <= 0.8)):

        return '0.7-0.8'

    elif ((merge['proba'] > 0.8) and (merge['proba']  <= 0.9)):

        return '0.8-0.9'

    elif merge['proba'] > 0.9 :

        return '0.9-1.0'

    else:

        return 0

 

#%%セグメントした条件を元データに当てはめて新しいカラムを追加する

merge2['proba_range'] = merge2.apply(pivot,axis= 1)

 

#%%add coupon proba

 

coupon_proba= pd.read_csv('D:/customer_coupon_Jun.csv',index_col =0)

 

coupon_proba2 = coupon_proba[['CUSTOMER_ID','proba']].set_index('CUSTOMER_ID').rename(columns={'proba': 'coupon_proba'})

coupon_proba2 =coupon_proba2['coupon_proba'].apply(lambda x:1 if x >=0.5  else 0 ).reset_index()

#%%add coupon proba

 

merge2= merge2.merge(coupon_proba2, how = 'left', on = 'CUSTOMER_ID')

merge2.iloc[0]

#%%購入適合確率ごとのセグメント毎の売上げ、購入回数、リセンシーを算出する

grouped=merge2.groupby(['proba_range','coupon_proba'])

 

grouped =grouped.agg({

       'proba_range':  'count',

     

       'RESPONSE': ['sum'],

       'G_AMT_1MONTH': ['sum'],

       'G_AMT_3MONTH': ['sum'],

       'G_AMT_6MONTH': ['sum'],

       'CHURN_3MONTH': ['sum'],

       'CHURN_6MONTH': ['sum'],

      }).round(1).sort_index(ascending=False)

 

grouped.to_excel(excel_writer = "D:/segment_info.xlsx",

                      sheet_name="segment",

                          startrow=1,

                          startcol=1)


 

 

 

 

 

 

 

 

 

 

#%%Ｈｏｌｄ－ｏｕｔ　ｓｅｔ（あたらしいデータ）をiterationで1万ごとのデータチャンクごとに標準化しながら予測モデルを当てはめて予測する

 

# Create empty list: dfs

customer = []

score = []

 

# Loop over 'WDI.csv'

for chunk in pd.read_csv('D:/CHURN_HOLDOUT_ALL.csv',index_col = 0, chunksize=10000):

   

    df_flag = chunk.max().reset_index()

    df_reordered = df_flag['index'].str.contains('Under40|Age|Over80|flg').rename('flg')

    df_flag2 = pd.concat([df_flag,df_reordered], axis = 1)

    list_1_original = df_flag2[df_flag2.flg == True]

    list_1 = list_1_original['index'].tolist()

 

    df_num = chunk.max().reset_index()

    df_reordered2 = df_num['index'].str.contains('sales|pen|cnt|share|s_|age|recency|YEAR1|ITVL').rename('flg')

    df_num2 = pd.concat([df_num,df_reordered2], axis = 1)

    list_2_original = df_num2[df_num2.flg == True]

    list_2 = list_2_original['index'].tolist()

 

 

    x_1 = chunk.drop(list_1,axis = 1 ).fillna(0)

    x_2 = chunk.drop(list_2,axis = 1 ).fillna(0)

 

    # それぞれのデータから目的変数を抜く

    x_1 = x_1.drop('RESPONSE' ,axis = 1 )

    x_2 = x_2.drop('RESPONSE' ,axis = 1 )

 

    # standardization

    data_1_scaled = StandardScaler().fit_transform(x_1)

    

    data_1 = pd.DataFrame(data_1_scaled, columns=x_1.columns,index=x_1.index)

 

    # concatenate features

    data_2 = pd.concat([data_1,x_2],axis=1)

   

    # Create the first Series

    result = gbrt.predict_proba(data_2)

    result = pd.DataFrame(result)

    result = result.iloc[:,1]

   

 

    # Create the second Series

    customer_id = chunk.loc[:,['app_sales','app_share','ec_sales','ec_pen','tsr_pen','ivr_pen']]

 

   

 

    # Append the filtered chunk to the list dfs

    customer.append(customer_id)

    score.append(result)

#%%year1のデータを格納する

 

df= pd.read_csv('D:/CHURN_HOLDOUT.csv',index_col =0)

year1 = df.G_AMT.reset_index()

 

 

 

#%%顧客後との予測数値をＣＳＶで吐き出す

customer_pd=pd.concat(customer).reset_index()

 

score_pd=pd.concat(score)

score_pd=pd.DataFrame(score_pd).reset_index(drop=True)

 

merge = customer_pd.join(score_pd, how='inner')

merge = merge.rename(columns={1: 'proba'})

merge = merge.merge(year1, how = 'inner', on = 'CUSTOMER_ID')

merge.to_csv("D:/customer.csv")

 

#%%予測適合確率ごとに顧客をセグメントする

 

def pivot(merge):

    if ((merge['proba'] >= 0) and (merge['proba']  <= 0.1)):

        return '0-0.1'

    elif ((merge['proba'] > 0.1) and (merge['proba']  <= 0.2)):

        return '0.1-0.2'

    elif ((merge['proba'] > 0.2) and (merge['proba']  <= 0.3)):

        return '0.2-0.3'

    elif ((merge['proba'] > 0.3) and (merge['proba']  <= 0.4)):

        return '0.3-0.4'

    elif ((merge['proba'] > 0.4) and (merge['proba']  <= 0.5)):

        return '0.4-0.5'

    elif ((merge['proba'] > 0.5) and (merge['proba']  <= 0.6)):

        return '0.5-0.6'

    elif ((merge['proba'] > 0.6) and (merge['proba']  <= 0.7)):

        return '0.6-0.7'

    elif ((merge['proba'] > 0.7) and (merge['proba']  <= 0.8)):

        return '0.7-0.8'

    elif ((merge['proba'] > 0.8) and (merge['proba']  <= 0.9)):

        return '0.8-0.9'

    elif merge['proba'] > 0.9 :

        return '0.9-1.0'

    else:

        return 0

 

#%%セグメントした条件を元データに当てはめて新しいカラムを追加する

merge['proba_range'] = merge.apply(pivot,axis= 1)

 

#%%購入適合確率ごとのセグメント毎の売上げ、購入回数、リセンシーを算出する

grouped=merge.groupby(['proba_range'])

 

grouped =grouped.agg({

       'proba_range':  'count',

     

       'G_AMT': ['sum','mean','median','std'],

       'ec_sales': ['mean','median','std'],

      }).round(1).sort_index(ascending=False)

 

grouped.to_excel(excel_writer = "D:/segment_info.xlsx",

                      sheet_name="segment",

                          startrow=1,

                          startcol=1)

#%%Hyperparameter tuning --randomizedsearchCV

from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV

 

param_dist = {"max_depth":[10,9,8,7,6,5,4,3,2,1,None],

              "learning_rate":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]}

 

gbrt = GradientBoostingClassifier()

 

gbrt_cv = RandomizedSearchCV(gbrt,param_dist,cv=5)

 

gbrt_cv.fit(X_train,y_train)

 

print("Tuned Decision Tree Parameters:{}".format(gbrt_cv.best_params_))

print("Best score is {}".format(gbrt_cv.best_score_))

#%%

gbrt.get_params().keys()

 

#%%Hyperparameter tuning --GridsearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

 

steps = [('GBRT', GradientBoostingClassifier())]

 

pipeline = Pipeline(steps)

 

pipeline.fit(X_train,y_train)

print("Test score:{:.2f}".format(pipeline.score(X_test,y_test)))

 

param_grid = {"GBRT__max_depth":[1,3,5,7,9,None],

              "GBRT__learning_rate":[0.01,0.03,0.05,0.07,0.09]}

 

grid = GridSearchCV(pipeline,param_grid=param_grid,cv=5)

grid.fit(X_train,y_train)

 

print("Best cross-validation accuracy:{:.2f}".format(grid.best_score_))

print("Test set score:{:.2f}".format(grid.score(X_test,y_test)))

print("Best parameters:{}".format(grid.best_params_))

#%% ROC_courve

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

 

y_pred_prob = gbrt.predict_proba(X_test)[:,1]

 

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

 

plt.plot([0,1],[0,1], 'k--')

plt.plot(fpr,tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
