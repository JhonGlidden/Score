import streamlit as st
import sys
sys.path.append('src/data')
from train_data import *
sys.path.append('src/models')
from predict_model import *



# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Scoring de Crédito", layout="wide")
st.image("https://th.bing.com/th/id/OIP.5Siyr_rfb3o-7qBqo07-oAHaFP?pid=ImgDet&rs=1", width=200)
st.title("Scoring de Crédito")

c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%
