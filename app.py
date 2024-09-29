#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

#######################
# Page configuration
st.set_page_config(
    page_title="Prediksi Kesenjangan Upah Gender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
""", unsafe_allow_html=True)

#######################
# Load data
df = pd.read_csv('gendergapinaverage new.csv')

#######################
# Sidebar
with st.sidebar:
    st.title('💼 Prediksi Kesenjangan Upah Gender')
    
    # Input untuk tahun dan negara
    year_input = st.number_input("Masukkan tahun", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), step=1)
    country_input = st.selectbox("Pilih negara", df['country'].unique())

#######################
# Preprocessing: Encode negara
label_encoder = LabelEncoder()
df['Country_encoded'] = label_encoder.fit_transform(df['country'])

# Memisahkan fitur dan target
X = df[['Country_encoded', 'Year']]
y = df['Gender wage gap %']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan fit model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#######################
# Dashboard Main Panel
col1, col2 = st.columns(2)

with col1:
    st.markdown('### Hasil Evaluasi Model')
    st.write(f"Mean Squared Error: {mse:.3f}")
    st.write(f"R² Score: {r2:.3f}")

with col2:
    # Visualisasi prediksi vs nilai aktual
    st.markdown('### Prediksi vs Nilai Aktual')
    lr_diff = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred})
    st.line_chart(lr_diff)

# Tombol prediksi
if st.button("Prediksi Kesenjangan Upah"):
    if year_input > 0 and country_input:  # Pastikan input valid
        # Melakukan prediksi kesenjangan upah berdasarkan input tahun dan negara
        country_encoded = label_encoder.transform([country_input])[0]
        input_data = pd.DataFrame([[country_encoded, year_input]], columns=['Country_encoded', 'Year'])
        predicted_gap = model.predict(input_data)[0]
        st.markdown(f"### Kesenjangan Upah yang Diprediksi untuk {country_input} pada tahun {year_input}: **{predicted_gap:.2f}%**")
    else:
        st.warning("Silakan masukkan tahun dan negara yang valid.")
