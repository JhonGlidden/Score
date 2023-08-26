import streamlit as st
import pandas as pd
import joblib
import io
import sys
import xlsxwriter
sys.path.append('src/data')
from train_data import *
sys.path.append('src/models')
from predict_model import Predict_model_App



# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Scoring de Crédito", layout="wide")
st.image("https://th.bing.com/th/id/OIP.5Siyr_rfb3o-7qBqo07-oAHaFP?pid=ImgDet&rs=1", width=200)
st.title("Scoring de Crédito")

c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%

with c30:
    uploaded_file=st.file_uploader("Escoja un archivo .txt", type='txt'                      
    )
    if uploaded_file is not None:
        info_box_wait = st.info(
            f"""
                Realizando la clasificación...
                """)
        #prediccion de la data de entrada 
        dato=pd.read_csv(uploaded_file, delimiter='\t')
        dato=Predict_model_App(dato)
        dato.one_encoding_transform()
        dato.load_model()
        dato.new_predict()
        dato.save_predict()
        dato=dato.output

        # csv = dato.to_csv(index=False)
        # csv_bytes = csv.encode('utf-8')
        # file_obj = io.BytesIO(csv_bytes)

        # # Crea un botón de descarga en Streamlit
        # st.download_button(
        #     label="Descargar CSV",
        #     data=file_obj,
        #     file_name="resultado.csv",
        #     mime="text/csv",
        # )


                
        # def to_excel(df):
        #     output = io.BytesIO()
        #     writer = pd.ExcelWriter(output, engine='xlsxwriter')
        #     df.to_excel(writer, sheet_name='Sheet1')
        #     writer.save()
        #     excel_data = output.getvalue()
        #     return excel_data
        
        def to_excel(df):
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sheet1')
            writer.close()  # Utiliza .close() en lugar de .save()
            output.seek(0)  # Asegúrate de posicionar el cursor al inicio antes de leer
            excel_data = output.getvalue()
            return excel_data


        if st.button('Transformando a un archivo Excel'):
            excel_data = to_excel(dato)

    # Cuando el usuario haga clic en el botón, se proporcionará una opción de descarga
            st.download_button(
            label="Descargar archivo Excel",
            data=excel_data,
            file_name="Score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )




    else:
        st.info(
            f"""
                👆 Debe cargar primero un archivo de datos con extensión .txt
                """
        )

        st.stop()