import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st


X_mod = pd.read_csv("data/dataset_garam(oversampled).csv")
X = X_mod.drop(columns=['Grade'])
y = X_mod['Grade']

def gnbclassifier():
    X_train_gnb, X_test_gnb, y_train_gnb, y_test_gnb = train_test_split(X, y, test_size=0.1, random_state=1)
    
    best_gnbclassifier = GaussianNB()
    best_gnbclassifier.fit(X_train_gnb, y_train_gnb)
    Y_pred_nb = best_gnbclassifier.predict(X_test_gnb)
    cm_bestgnb = confusion_matrix(y_test_gnb, Y_pred_nb)

    ac_bestgnb = round(accuracy_score(y_test_gnb, Y_pred_nb) * 100)
    return cm_bestgnb, ac_bestgnb

gnbclassifier()

def knn():
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.1, random_state=1)
    # st.write("Jumlah X Training", X_train_knn.shape)
    # st.write("Jumlah X Test", X_test_knn.shape)
    # st.write("Jumlah Y Train", y_train_knn.shape)
    # st.write("Jumlah Y Test", y_test_knn.shape)
    # st.write()

    best_knnclassifier = KNeighborsClassifier(n_neighbors=3)
    best_knnclassifier.fit(X_train_knn, y_train_knn)
    Y_pred_knn = best_knnclassifier.predict(X_test_knn)

    cm_bestknn = confusion_matrix(y_test_knn, Y_pred_knn)

    ac_bestknn = round(accuracy_score(y_test_knn, Y_pred_knn) * 100)
    # st.write(classification_report(y_test_knn, Y_pred_knn))
    return cm_bestknn, ac_bestknn
knn()

def dt():
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, test_size=0.1, random_state=1)
    best_dtclassifier = DecisionTreeClassifier()
    best_dtclassifier.fit(X_train_dt, y_train_dt)
    Y_pred_dt = best_dtclassifier.predict(X_test_dt)

    cm_bestdt = confusion_matrix(y_test_dt, Y_pred_dt)
    ac_bestdt = round(accuracy_score(y_test_dt, Y_pred_dt) * 100)
    return cm_bestdt, ac_bestdt
dt()

def svm():
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.1, random_state=1)
    best_svm_linear = SVC(kernel='linear')
    best_svm_linear.fit(X_train_svm, y_train_svm)
    Y_pred_svm_linear = best_svm_linear.predict(X_test_svm)

    cm_bestsvmlinear = confusion_matrix(y_test_svm, Y_pred_svm_linear)
    ac_bestsvmlinear = round(accuracy_score(y_test_svm, Y_pred_svm_linear) * 100)
    return cm_bestsvmlinear, ac_bestsvmlinear
svm()
