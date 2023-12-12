import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import time
import altair as alt

# from proses import preprocessing
from proses.preprocessing import balance
from proses import model
from proses.model import knn

from proses import implementasi


st.markdown("# Klasifikasi Data Garam")

st.markdown("# Pengolahan Data")
selected = option_menu(
    options=["Dataset", "Preprocessing", "Modeling", "Implementation"],
    orientation="horizontal",
)

X = pd.read_csv("data/dataset_garam.csv")
y = X['Grade']
if (selected == "Dataset"):
    st.success(
        f"Jumlah Data : {X.shape[0]} Data, dan Jumlah Fitur : {X.shape[1]} Fitur")
    dataframe, keterangan = st.tabs(['Dataset', 'Keterangan'])
    with dataframe:
        st.write(X)

    with keterangan:
        st.text("""
             Column:
             - Kadar air : Kadar air mengacu pada jumlah atau persentase air yang terkandung dalam suatu zat atau lingkungan tertentu
             - Tak larut : Tak larut adalah sifat suatu zat yang tidak dapat larut atau larut dengan sangat sedikit dalam pelarut tertentu pada suhu dan kondisi tertentu.
             - Kalsium : Kalsium adalah unsur kimia dengan simbol Ca dan nomor atom 20. Ini adalah logam alkali tanah yang termasuk dalam kelompok 2 tabel periodik. 
             - Magnesium : Magnesium adalah unsur kimia dengan simbol Mg dan nomor atom 12
             - Sulfat : Sulfat adalah ion negatif (anion) yang terdiri dari satu atom sulfur dan empat atom oksigen
             - NaCl(wb) : NaCl (wb) adalah singkatan dari "Natrium Klorida (basis berat)" atau "Sodium Chloride (wet basis)" dalam bahasa Inggris
             - NaCl(db): NaCl (db) adalah singkatan dari "Natrium Klorida (dry basis)" atau "Sodium Chloride (dry basis)"
             
             Label
             Output Dari Dataset ini yaitu K1 K2 K3 dan K4
           """)

########################################## Preprocessing #####################################################
elif (selected == 'Preprocessing'):
    balance()


elif selected == 'Modeling':
    nb = st.tabs(['Naive Bayes'])
    with nb:
        st.markdown("# Algoritma Naive Bayes")
        # Menangkap Confusion Matrix dan akurasi yang dikembalikan
        cm_bestgnb, ac_bestgnb = model.gnbclassifier()

        # st.write("Confusion Matrix:")
        # st.write(cm_bestgnb)

        st.success("Akurasi Naive Bayes Gaussian dengan split dataset 90:10 adalah : " + str(ac_bestgnb) + "%")
    
    # with knn:
    #     st.markdown("# Algoritma K-Nearest Neighbor")
    #     cm_bestknn, ac_bestknn = model.knn()

    #     # st.write("Confusion Matrix:")
    #     # st.write(cm_bestknn)

    #     st.success("Akurasi K-Nearest Neighbor dengan K : 3 split dataset 90:10 adalah : " + str(ac_bestknn) + "%")
    
    # with dt:
    #     st.markdown("# Algoritma Decision Tree")
    #     cm_bestdt, ac_bestdt = model.dt()

    #     # st.write("Confusion Matrix:")
    #     # st.write(cm_bestdt)

    #     st.success("Akurasi Decision Tree split dataset 90:10 adalah : " + str(ac_bestdt) + "%")

    # with svm:
    #     st.markdown("# Algoritma Support Vector Machine")
    #     cm_bestsvmlinear, ac_bestsvmlinear = model.svm()

    #     # st.write("Confusion Matrix:")
    #     # st.write(cm_bestdt)

        # st.success("Akurasi Support Vector Machine split dataset 90:10 adalah : " + str(ac_bestsvmlinear) + "%")
    with grafik:
         st.write ("##### Grafik Akurasi Semua Model") 
         data = pd.DataFrame({
            'Akurasi' : [ac_bestgnb],
            'Model' : ['Naive Bayes'],
            })

         chart = (
             alt.Chart(data)
             .mark_bar()
             .encode(
                alt.X("Akurasi"),
                alt.Y("Model"),
                # alt.Color("Akurasi"),
                alt.Color("Model", scale=alt.Scale(scheme='category10')),
                alt.Tooltip(["Akurasi", "Model"]),
            )
            .interactive()
            )
         st.altair_chart(chart,use_container_width=True, theme="streamlit")
####################### Implementasi ############################


elif selected == 'Implementation':
    Kadar_air = st.number_input('Input nilai Kadar Air')
    Tak_larut = st.number_input('Input nilai Tak Larut')
    Kalsium = st.number_input('Input nilai Kalsium')
    Magnesium = st.number_input('Input nilai Magnesium')
    Sulfat = st.number_input('Input nilai Sulfat')
    NaCl_wb = st.number_input('Input nilai NaCl (wb)')
    NaCl_db = st.number_input('Input nilai NaCl (db)')

    # Melakukan normalisasi pada data input
    data_input = [[Kadar_air, Tak_larut, Kalsium,
                   Magnesium, Sulfat, NaCl_wb, NaCl_db]]

    # Memuat model
    # scaler = joblib.load('model/df_scaled(norm).save')
    # data_input_scaled = scaler.transform(data_input)
    nb = st.tabs(['Naive Bayes'])
    with nb:
        grade_prediction_nb = implementasi.nb(data_input)
        # st.write("Hasil Prediksi Naive Bayes : ", grade_prediction_nb)
    # with knn:
    #     grade_prediction_knn = implementasi.knn(data_input)
    #     # st.write("Hasil Prediksi K-Nearest Neighbor : ", grade_prediction_knn)
    # with dt:
    #     grade_prediction_dt = implementasi.dt(data_input)
    #     # st.write("Hasil Prediksi Decision Tree: ", grade_prediction_dt)
    # # with svm:
    #     grade_prediction_svm = implementasi.svm(data_input)
    #     # st.write("Hasil Prediksi Support Vector Machine: ", grade_prediction_svm)
    #     # implementasi.nb(data_input)
    #     # st.write("Hasil Prediksi")
    # button = st.button('Prediksi')
    # if button:
    #     grade_prediction = implementasi.nb(data_input)
    #     # st.write("Hasil Prediksi: ", grade_prediction)
