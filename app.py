import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import plotly.graph_objs as go
import os

from wordcloud import WordCloud
from collections import Counter
from streamlit_option_menu import option_menu
from pymongo import MongoClient
from dotenv import load_dotenv

# Settings
st.set_page_config(page_title="Analisis Sentimen")

st.markdown("""
    <style>
        div.block-container {padding-top:2rem;}
        div.block-container {padding-bottom:2rem;}
    </style>
""", unsafe_allow_html=True)

# MongoDB Connection
load_dotenv(".env")
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
db = client[os.getenv("MONGO_DATABASE_NAME")]
collection = db[os.getenv("MONGO_COLLECTION_NAME")]

# Load Model
def load_model():
    return pickle.load(open('sentiment_analysis_model.pkl', 'rb'))

# Add Data
def add_data(username, access_control, name, password):
    data = {
        "username": username,
        "access_control": access_control,
        "name": name,
        "password": password
    }
    return collection.insert_one(data)

# Delete Data
def delete_data(username):
    return collection.delete_one({"username": username})

# Update User
def update_user(username, access_control, name, password):
    collection.update_one(
        {"username": username},
        {"$set": {
            "access_control": access_control,
            "name": name,
            "password": password
        }}
    )

# Login
def validate_login(username, password):
    user = collection.find_one({"username": username, "password": password})
    return user

# Cache Data
@st.cache_data
def load_users_data():
    users_data = list(collection.find({}, {"_id": 0, "username": 1, "access_control": 1, "name": 1}))
    return pd.DataFrame(users_data)

# Clear Cache
def clear_cache():
    load_users_data.clear()

# Halaman Login
def login_page():
    st.markdown("<h1 style='text-align: center;'>Selamat Datang!</h1>", unsafe_allow_html=True)
    st.text("")

    col1, col2, col3 = st.columns([0.5, 1.5, 0.5])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username Anda")
            password = st.text_input("Password", placeholder="Masukkan password Anda", type="password")
            login_button = st.form_submit_button("Login", type="primary")

        if login_button:
            user = validate_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Username atau password salah. Silakan coba lagi.")

# Halaman About
def about_page():
    st.title('Analisis Sentimen Komentar Youtube Honda Menggunakan Metode Naive Bayes')
    st.write("Halaman ini bertujuan untuk menyajikan hasil analisis sentimen komentar pada kanal YouTube **'Welove Honda Indonesia'** pada video yang berjudul **'Klarifikasi Kemunculan Warna Kuning Pada Rangka Honda'**. Data diambil dari komentar netizen, dengan total data setelah dilakukan preprocessing mencapai **2240 data**. Kami melakukan pelabelan otomatis menggunakan metode **Lexicon Based** yang membagi data menjadi dua kelas, yaitu **Positif dan Negatif**.")
    
    # Membaca file Excel
    data = pd.read_excel('hasil_labeling.xlsx')
    st.write("**Data Labeling**:")
    st.dataframe(data, use_container_width=True)

    # Mendapatkan jumlah Label Positif, Netral, dan Negatif
    label_counts = data['sentiment_label'].value_counts()

    # Menghitung jumlah sentimen positif dan negatif
    positive_count = data[data['sentiment_label'] == 'positive'].shape[0]
    negative_count = data[data['sentiment_label'] == 'negative'].shape[0]

    # Membuat label dan nilai baru
    new_label_counts = pd.concat([label_counts, pd.Series({'positive': positive_count, 'negative': negative_count})])

    # Visualisasi Grafik Pie
    st.subheader('Grafik Pie untuk Sentimen Label')
    fig_pie = go.Figure(data=[go.Pie(labels=label_counts.index, values=label_counts.values)])
    st.plotly_chart(fig_pie, use_container_width=True)

    # Visualisasi Grafik Batang
    st.subheader('Grafik Batang untuk Sentimen Label')
    fig_bar = go.Figure(data=[go.Bar(x=label_counts.index, y=label_counts.values)])
    fig_bar.update_xaxes(title='Sentiment Label')
    fig_bar.update_yaxes(title='Count')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Filter teks berdasarkan sentimen
    positive_text = ' '.join(data[data['sentiment_label'] == 'Positif']['text'])
    negative_text = ' '.join(data[data['sentiment_label'] == 'Negatif']['text'])
    all_text = ' '.join(data['text'])

    # Visualisasi Wordcloud untuk sentimen positif
    st.subheader('Wordcloud untuk Sentimen Positif')
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    st.image(wordcloud_positive.to_array(), caption='Wordcloud Sentimen Positif', use_column_width=True)

    # Visualisasi Wordcloud untuk sentimen negatif
    st.subheader('Wordcloud untuk Sentimen Negatif')
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    st.image(wordcloud_negative.to_array(), caption='Wordcloud Sentimen Negatif', use_column_width=True)

    # Visualisasi Wordcloud untuk semua kata
    st.subheader('Wordcloud untuk Semua Kata')
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), caption='Wordcloud Semua Kata', use_column_width=True)

    # Sentimen positif
    positive_text_combined = ' '.join(data[data['sentiment_label'] == 'Positif']['text'])
    positive_words = positive_text_combined.split()
    positive_word_freq = Counter(positive_words)
    top_20_positive_words = positive_word_freq.most_common(10)
    positive_words, positive_freq = zip(*top_20_positive_words)

    # Sentimen negatif
    negative_text_combined = ' '.join(data[data['sentiment_label'] == 'Negatif']['text'])
    negative_words = negative_text_combined.split()
    negative_word_freq = Counter(negative_words)
    top_20_negative_words = negative_word_freq.most_common(10)
    negative_words, negative_freq = zip(*top_20_negative_words)

    # Semua kata
    all_text_combined = ' '.join(data['text'])
    all_words = all_text_combined.split()
    all_word_freq = Counter(all_words)
    top_20_all_words = all_word_freq.most_common(10)
    all_words, all_freq = zip(*top_20_all_words)

    # Membuat grafik bar untuk sentimen positif
    fig_positive = go.Figure([go.Bar(x=positive_words, y=positive_freq, marker=dict(color='green'))])
    fig_positive.update_layout(title='10 Kata yang Paling Sering Muncul (Sentimen Positif)', xaxis_title='Kata', yaxis_title='Frekuensi')

    # Membuat grafik bar untuk sentimen negatif
    fig_negative = go.Figure([go.Bar(x=negative_words, y=negative_freq, marker=dict(color='red'))])
    fig_negative.update_layout(title='10 Kata yang Paling Sering Muncul (Sentimen Negatif)', xaxis_title='Kata', yaxis_title='Frekuensi')

    # Membuat grafik bar untuk semua kata
    fig_all = go.Figure([go.Bar(x=all_words, y=all_freq, marker=dict(color='blue'))])
    fig_all.update_layout(title='10 Kata yang Paling Sering Muncul (Semua)', xaxis_title='Kata', yaxis_title='Frekuensi')

    # Menampilkan grafik menggunakan Plotly
    st.plotly_chart(fig_positive, use_container_width=True)
    st.plotly_chart(fig_negative, use_container_width=True)
    st.plotly_chart(fig_all, use_container_width=True)

    # Menambahkan kalimat tambahan
    st.markdown("""
        Setelah dilakukan pelabelan dengan lexicon based, selanjutnya kami melakukan feature extractor dengan **TF-IDF**. Selanjutnya kami melakukan klasifikasi dengan **Complement Naive Bayes**. Berikut adalah hasil evaluasi:

        - **Akurasi Complement Naive Bayes:** 84.82%
        - **Presisi Complement Naive Bayes:** 84.91%
        - **Recall Complement Naive Bayes:** 84.82%
        - **F1-score Complement Naive Bayes:** 84.69%
    """)

# Halaman Predict Text
def predict_text_page(model):
    st.subheader("Predict Sentiment from Text")
    tweet = st.text_input('Enter your tweet')
    submit = st.button('Predict')

    if submit:
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
        st.write('Predicted Sentiment:', prediction[0])

# Halaman Predict DataFrame
def predict_dataframe_page(model):
    st.subheader("Predict Sentiment from DataFrame")
    uploaded_file = st.file_uploader("Upload your DataFrame", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' else pd.read_csv(uploaded_file)
        st.write('Uploaded DataFrame:')
        st.write(df)

        submit_df = st.button('Predict Sentiments from DataFrame')

        if submit_df:
            start = time.time()
            predictions = model.predict(df['text'])
            end = time.time()
            st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
            st.write('Predicted Sentiments:')
            st.write(predictions)

            # Menambahkan grafik pie
            sentiment_counts = pd.Series(predictions).value_counts()
            st.subheader('Sentiment Distribution')

            # Membuat data untuk grafik pie
            labels = sentiment_counts.index
            values = sentiment_counts.values

            # Menentukan palet warna
            palette_colors = ['#be185d', '#500724']

            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values,  marker=dict(colors=palette_colors))])
            st.plotly_chart(fig_pie, use_container_width=True)

            # Menambahkan grafik bar
            st.subheader('Sentiment Counts')

            fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=palette_colors)])
            fig_bar.update_layout(xaxis_title='Sentiment', yaxis_title='Count')
            st.plotly_chart(fig_bar, use_container_width=True)

# Halaman Access Management Admin
def access_management_page_admin():
    with st.sidebar:
        access_management_page_option = st.selectbox("Menu Access Management", ["Add User", "Delete User", "Edit User"])

    users_data = load_users_data()

    with st.container(border=True):
        st.subheader("Users Data")
        users_df = st.dataframe(users_data, use_container_width=True)

    st.divider()

    if access_management_page_option == "Add User":
        with st.form("add_user_form", clear_on_submit=True):
            st.subheader("Form Add User")

            add_username = st.text_input("Username", placeholder="Masukkan username")
            add_access_control = st.selectbox("Access Control", ["Admin", "User"], placeholder="Pilih access control", index=None)
            add_name = st.text_input("Nama", placeholder="Masukkan nama")
            add_password = st.text_input("Password", placeholder="Masukkan password", type="password")
            add_user_button = st.form_submit_button("Add User", type="primary")

            if add_user_button:
                if add_username and add_access_control and add_name and add_password:
                    add_data(add_username, add_access_control, add_name, add_password)
                    message_success = st.success(f"User {add_name} telah berhasil ditambahkan")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap isi semua field")
                    time.sleep(3)
                    message_error.empty()

    elif access_management_page_option == "Delete User":
        with st.form("delete_user_form", clear_on_submit=True):
            st.subheader("Form Delete User")

            delete_username = st.selectbox("Username", np.sort(pd.DataFrame(users_data)["username"].unique()), placeholder="Pilih username", index=None)
            delete_user_button = st.form_submit_button("Delete User", type="primary")

            if delete_user_button:
                if delete_username:
                    delete_data(delete_username)
                    message_success = st.success(f"User {delete_username} telah berhasil dihapus")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap pilih username")
                    time.sleep(3)
                    message_error.empty()

    else:
        with st.form("edit_user_form", clear_on_submit=True):
            st.subheader("Form Edit User")

            edit_username = st.selectbox("Username", np.sort(pd.DataFrame(users_data)["username"].unique()), placeholder="Pilih username", index=None)
            edit_access_control = st.selectbox("Access Control", ["Admin", "User"], placeholder="Pilih access control", index=None)
            edit_name = st.text_input("Nama", placeholder="Masukkan nama")
            edit_password = st.text_input("Password", placeholder="Masukkan password", type="password")
            edit_user_button = st.form_submit_button("Edit User", type="primary")

            if edit_user_button:
                if edit_username and edit_access_control and edit_name and edit_password:
                    update_user(edit_username, edit_access_control, edit_name, edit_password)
                    message_success = st.success(f"User {edit_username} telah berhasil diperbarui")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap isi semua field")
                    time.sleep(3)
                    message_error.empty()

# Halaman Access Management User
def access_management_page_user():
    current_user = st.session_state.get("user", {})
    edit_username = current_user.get("username", "")
    edit_access_control = current_user.get("access_control", "User")

    with st.form("edit_user_form_user", clear_on_submit=True):
        st.subheader("Form Edit User")

        edit_name = st.text_input("Nama", placeholder="Masukkan nama")
        edit_password = st.text_input("Password", placeholder="Masukkan password", type="password")
        edit_user_button = st.form_submit_button("Edit User", type="primary")

        if edit_user_button:
            if edit_name and edit_password:
                update_user(edit_username, edit_access_control, edit_name, edit_password)
                message_success = st.success(f"User {edit_username} telah berhasil diperbarui")
                time.sleep(3)
                message_success.empty()

                clear_cache()
                st.session_state.user["name"] = edit_name
                st.rerun()
            else:
                message_error = st.error("Harap isi semua field")
                time.sleep(3)
                message_error.empty()

# Session State untuk Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Tampilkan Halaman Login Jika Belum Login
if not st.session_state.logged_in:
    login_page()
else:
    # Tampilkan Halaman Isi
    with st.sidebar:
        user = st.session_state.get("user", {})
        user_name = user.get("name", "User")
        user_access = user.get("access_control", "user")
        st.markdown(f"Selamat datang, {user_access} {user_name}")

        menu_title = "Menu Admin" if user_access == "Admin" else "Menu User"
        menu_options = ["About", "Predict Text", "Predict DataFrame", "Access Management"]
        option = option_menu(
            menu_title,
            menu_options,
            menu_icon="cast",
            default_index=0
        )

        logout_button = st.button("Logout", type="primary")

    # Logout
    if logout_button:
        st.session_state.logged_in = False
        st.rerun()

    # Menentukan Halaman yang Sesuai
    if option == "About":
        about_page()
    elif option == "Predict Text":
        model = load_model()
        predict_text_page(model)
    elif option == "Predict DataFrame":
        model = load_model()
        predict_dataframe_page(model)
    elif option == "Access Management":
        if user_access == "Admin":
            access_management_page_admin()
        else:
            access_management_page_user()
