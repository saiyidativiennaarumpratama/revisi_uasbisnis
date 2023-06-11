import streamlit as st
import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import RandomOverSampler


a = pd.read_csv("data/dataset_garam.csv")
X_awal = a.drop(['Grade'], axis=1)
Y_awal = a['Grade']

def balance():
  st.markdown("**Data Awal Sebelum di lakukan Imbalancing Data**")
  st.dataframe(a)
  st.write(f"Jumlah Data Sebelum Oversampling: {a.shape[0]} Data, dan Jumlah Fitur: {a.shape[1]} Fitur")
  
  # Memisahkan fitur dan target
  X_balance = X_awal.values
  y_balance = Y_awal
  # Menghitung jumlah sampel di setiap kelas
  jumlah_kelas = y_balance.value_counts().sort_index()

  # Menampilkan jumlah sampel di setiap kelas sebelum penyeimbangan
  st.write("Jumlah sampel sebelum penyeimbangan:")
  st.write(jumlah_kelas, '\n')

  # Inisialisasi RandomOverSampler untuk oversampling
  oversampler = RandomOverSampler()

  # Melakukan oversampling pada data
  X_oversampled, y_oversampled = oversampler.fit_resample(X_balance, y_balance)

  # Membuat dataframe hasil oversampling
  X_aftterbalance = pd.DataFrame(X_oversampled, columns=X_awal.columns)
  X_aftterbalance[Y_awal.name] = y_oversampled

  # Menampilkan jumlah sampel di setiap kelas setelah oversampling
  st.write("Jumlah sampel setelah oversampling:")
  st.write(X_aftterbalance[Y_awal.name].value_counts())

  st.markdown("**Data setelah dilakukan Imbalancing data dengan Oversampling**")
  st.write(X_aftterbalance)
  st.write(f"Jumlah Data Setelah Oversampling: {X_aftterbalance.shape[0]} Data, dan Jumlah Fitur: {X_aftterbalance.shape[1]} Fitur")

  # scaler = MinMaxScaler()
  # data_scaled = scaler.fit_transform(X_pre)

  # #memasukan fitur 
  # features_names = X_pre.columns.copy()
  # scaled_features = pd.DataFrame(data_scaled, columns=features_names)
  # st.write(scaled_features)

  # # Save Scaled
  # scaler_filename = "df_scaled(norm).save"
  # joblib.dump(scaler, scaler_filename)
 
