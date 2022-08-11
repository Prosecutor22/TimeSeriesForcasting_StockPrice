import requests
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
from PIL import Image


st.set_page_config(page_title="TimeSeriesForecasting", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----


# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi, we are Group 8 from Big Data Club:wave:")
    st.title("Time Series Forecasting - Stock Prediction")
    st.write(
        "This product is used for Stock Prediction."
    )
    st.write("[See our poster >](url here)")
    

# ---- ABOUT PROJECT ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("About our project")
        st.write("##")
        st.write("Will be add soon"
        )
        st.write("[See our report >](https://www.overleaf.com/read/zmjvcpqqtwfm)")
        st.write("[See our slide >](url here)")
    # with right_column:
    #     st_lottie(example, height=300, key="example")

# ---- IMPORT DATA ----
with st.container():
    st.write("---")
    st.header("Import data:")
    uploaded_file = st.file_uploader("Please import as CSV file")
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)


# ---- VISUALIZE DATA ----
with st.container():
    st.write("---")
    st.header("Data visualization")

# ---- MODEL PREDICTION ----
with st.container():
    st.write("---")
    st.header("Predict price")


# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Give us your feedback!")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/phanphuocminh2002@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()