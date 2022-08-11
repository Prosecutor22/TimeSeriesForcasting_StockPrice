import time
import requests
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from matplotlib.animation import FuncAnimation

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

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
flag = False
with st.container():
    st.write("---")
    st.header("Import data:")
    uploaded_file = st.file_uploader("Please import as CSV file")
    if uploaded_file is not None:
        st.write("Load data successful " + time.strftime("%H:%M:%S"))
        flag = True
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date').filter(['Price'])
        df = df.iloc[::-1]


# ---- VISUALIZE DATA ----
with st.container():
    st.write("---")
    st.header("Data visualization")
    if flag == True:
        if st.button("Click here to view data"):
            st.write(df) 
        if st.button("Click here to visualize your data"):
            fig = plt.figure(figsize=(16,6))
            plt.title('Price History')
            plt.plot(df['Price'])
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price x1000 VND', fontsize=18)
            st.pyplot(fig)
        
        

# ---- MODEL PREDICTION ----
with st.container():
    st.write("---")
    st.header("Predict price")
    if flag == True:
        if st.button("Click here to start model"):
            # ---- DATA PROCESSING ---- 
            dataset = df.values
            training_data_len = int(np.ceil(len(dataset) * .9))
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)
            train_data = scaled_data[0:int(training_data_len), :]
            x_train = []
            y_train = []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # ---- BUILD MODEL ---- 
            from keras.layers import SimpleRNN
            model = Sequential()
            model.add(SimpleRNN(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(SimpleRNN(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # ---- TRAIN MODEL ---- 
            model.fit(x_train, y_train, batch_size=1, epochs=1)
            test_data = scaled_data[training_data_len - 60: , :]
            
            # ---- EVALUATE ---- 
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])
                
            x_test = np.array(x_test)

            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

            train = df[:training_data_len]
            valid = df[training_data_len:]
            valid['Predictions'] = predictions

            predict_fig = plt.figure(figsize=(16,6))
            plt.title('Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price x1000 VND', fontsize=18)
            plt.plot(train['Price'])
            plt.plot(valid[['Price', 'Predictions']])
            plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
            st.pyplot(predict_fig)


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