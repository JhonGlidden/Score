import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('./src/data')
from clean_data import Clean_data 
sys.path.append('./src/data')
from train_data import Train_data 


'''
Vamos a predecir para nueva data, el problema radica en qué psa si la nueva data no tiene una categoria con la cual entrenamos el modelo, cuando hacemos el encoding
no aparecerá esa categoria. Aqui solucionamos este problema 
'''


class Predict_model:
    '''
    Vamos a predecir para nueva data, el problema radica en qué psa si la nueva data no tiene una categoria con la cual entrenamos el modelo, cuando hacemos el encoding
    no aparecerá esa categoria. Aqui solucionamos este problema 
    '''

    def __init__(self, new_data):
        self.new_data = new_data
        #self.output_path = output_path
    def one_encoding_transform(self):
        '''
        Lectura 
        Limpieza
        '''
        # convertir el dataframe a una lista de nombres de columnas
        names_one_hot_encoding = pd.read_csv('./data/output/cat.csv', header=None)
        names_one_hot_encoding = names_one_hot_encoding[0].tolist()
        names_one_hot_encoding.remove('MarcaMora_Tarjeta')

        data_input=Train_data(self.new_data)
        #data_input.read_data()
        data_input.clean_data()
        data_input.delete_col('MarcaMora_Tarjeta')
        data_input.delete_col('Fecha')
        data_input.delete_col('CODIGO_ID')
        data_input.delete_col('SALDO_TOTAL_TARJETA')
        data_input.one_hot_encoding()
        data_input=data_input.df.reindex(columns=names_one_hot_encoding, fill_value=0)
        self.df=data_input

    def load_model(self):
        self.modelo_random_forest = joblib.load('./model/modelo_entrenado.joblib')

    def new_predict(self):
        '''
        Lectura 
        Limpieza
        '''
        self.value_predict=self.modelo_random_forest.predict(self.df)
        self.value_predict=self.value_predict.tolist()
        self.prob_predict=self.modelo_random_forest.predict_proba(self.df)
        self.prob_predict=self.prob_predict.tolist()

    def save_predict(self,output_path):
        data_output=Train_data(self.new_data)
        #data_output.read_data()
        self.output=data_output.df
        self.output['PREDICCION']=self.value_predict
        self.output['PROBABILIDAD']=self.prob_predict
        self.output.to_csv(output_path,index=False)

#################################################################################################################################################################################

class Predict_model_App:
    '''
    Vamos a predecir para nueva data, el problema radica en qué psa si la nueva data no tiene una categoria con la cual entrenamos el modelo, cuando hacemos el encoding
    no aparecerá esa categoria. Aqui solucionamos este problema 
    '''

    def __init__(self, new_data):
        self.new_data = new_data
        #self.output_path = output_path

    def one_encoding_transform(self):
        names_one_hot_encoding = pd.read_csv('./data/output/cat.csv', header=None)
        names_one_hot_encoding = names_one_hot_encoding[0].tolist()
        names_one_hot_encoding.remove('MarcaMora_Tarjeta')

        data_input=Train_data(self.new_data)
        data_input.clean_data()
        data_input.delete_col('MarcaMora_Tarjeta')
        data_input.delete_col('Fecha')
        data_input.delete_col('CODIGO_ID')
        data_input.delete_col('SALDO_TOTAL_TARJETA')
        data_input.one_hot_encoding()
        data_input=data_input.df.reindex(columns=names_one_hot_encoding, fill_value=0)
        self.df=data_input

    def load_model(self):
        self.modelo_random_forest = joblib.load('./model/modelo_entrenado.joblib')

    def new_predict(self):
        '''
        Lectura 
        Limpieza
        '''
        self.value_predict=self.modelo_random_forest.predict(self.df)
        self.value_predict=self.value_predict.tolist()
        self.prob_predict=self.modelo_random_forest.predict_proba(self.df)[:,1]
        self.prob_predict=self.prob_predict.tolist()

    def save_predict(self):
        data_output=Train_data(self.new_data)
        #data_output.read_data()
        self.output=data_output.df
        self.output['PREDICCION']=self.value_predict
        self.output['PROBABILIDAD']=self.prob_predict
        #self.output.to_csv(output_path,index=False)



# entrada=pd.read_csv("./data/raw/BasePrueba.txt",delimiter='\t')
# x=Predict_model_App(entrada)
# x.one_encoding_transform()
# x.load_model()
# x.new_predict()
# x.save_predict()
# print(x.output)

# ==h[^=]+$