import streamlit as st
import joblib

Grade = [0, 1, 2, 3]


def nb(data_input_baru):
    modelnb = joblib.load('model/bestmodelNB.pkl')

    st.write('Hasil Prediksi Menggunakan Algoritma Naive Bayes')
    Y_pred_nb = modelnb.predict(data_input_baru)
    prediction_index = int(Y_pred_nb[0][-1])

    st.success(
        f'Berdasarkan data yang sudah diinputkan, termasuk dalam kelas: {Grade[prediction_index-1]}')
    
def knn(data_input_baru):
    modelknn = joblib.load('model/bestmodelKNN.pkl')

    st.write('Hasil Prediksi Menggunakan Algoritma K-Nearest Neighbor')
    Y_pred_knn = modelknn.predict(data_input_baru)
    prediction_index = int(Y_pred_knn[0][-1])

    st.success(
        f'Berdasarkan data yang sudah diinputkan, termasuk dalam kelas: {Grade[prediction_index-1]}')

def dt(data_input_baru):
    modeldt = joblib.load('model/bestmodelDT.pkl')

    st.write('Hasil Prediksi Menggunakan Algoritma Decision Tree')
    Y_pred_dt = modeldt.predict(data_input_baru)
    prediction_index = int(Y_pred_dt[0][-1])

    st.success(
        f'Berdasarkan data yang sudah diinputkan, termasuk dalam kelas: {Grade[prediction_index-1]}')

def svm(data_input_baru):
    modelsvm = joblib.load('model/bestmodelSVMLinear.pkl')

    st.write('Hasil Prediksi Menggunakan Algoritma Support Vector Machine')
    Y_pred_svm = modelsvm.predict(data_input_baru)
    prediction_index = int(Y_pred_svm[0][-1])

    st.success(
        f'Berdasarkan data yang sudah diinputkan, termasuk dalam kelas: {Grade[prediction_index-1]}')    

