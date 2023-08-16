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


# # Cargar el modelo desde el archivo

# modelo_random_forest = joblib.load('./model/modelo_entrenado.joblib')

# data_entrada=pd.read_csv("./data/raw/BasePruebaAval.txt",delimiter='\t')


# ############################################
# df_model=Train_data("./data/raw/BasePruebaAval.txt")
# df_model.read_data()
# ## Limpiamos la data , el tipo de datos y reemplazamos si por 1 y no por 0
# df_model.clean_data()
# #E Eliminamos las columnas que no nos sirven para modelado
# df_model.delete_col('MarcaMora_Tarjeta')
# df_model.delete_col('Fecha')
# df_model.delete_col('CODIGO_ID')
# df_model.delete_col('SALDO_TOTAL_TARJETA')

# ## Creamos un encoding para las categorias 
# df_model.one_hot_encoding()
# df_new_predict=df_model.df

# #names_one_hot_encoding=np.load("./data/output/cat.npy")


# names_one_hot_encoding = pd.read_csv('./data/output/cat.csv', header=None)

# # convertir el dataframe a una lista de nombres de columnas

# names_one_hot_encoding = names_one_hot_encoding[0].tolist()
# names_one_hot_encoding.remove('MarcaMora_Tarjeta')
# #print(names_one_hot_encoding)
# df_new_predict = df_new_predict.reindex(columns=names_one_hot_encoding, fill_value=0)

#print(df_new_predict)

#names_one_hot_encoding=names_one_hot_encoding.tolist()
#x_hot=pd.DataFrame(columns=names_one_hot_encoding,index=range(len(df_model)))

#print(x_hot)






# for i in x_names_one_hot_encoding:
#     x_hot.loc[:,rf'{i}']=df_model[rf'{i}'].values

# x_hot=x_hot.fillna(value=False)
# x_hot=x_hot.drop([ 'INSTRUCCION_TEC','SUCURSAL_MACHALA','SEGMENTO_RIESGO_C'],axis=1)

# x_hot=x_hot.drop('MarcaMora_Tarjeta',axis=1)



# # ## Cargo el modelo
# prediccion=modelo_random_forest.predict(df_new_predict)
# prediccion=prediccion.tolist()
# probabilidad=modelo_random_forest.predict_proba(df_new_predict)
# probabilidad=probabilidad.tolist()
# data_entrada['PREDICCION']=prediccion
# data_entrada['PROBABILIDAD']=probabilidad
# ## Exporto los resultados
# data_entrada.to_csv('./data/output/data_salida.csv',index=False)

class Predict_model:
    '''
    Vamos a predecir para nueva data, el problema radica en qué psa si la nueva data no tiene una categoria con la cual entrenamos el modelo, cuando hacemos el encoding
    no aparecerá esa categoria. Aqui solucionamos este problema 
    '''

    def __init__(self, new_data_path,output_path):
        self.new_data_path = new_data_path
        self.output_path = output_path

    def one_encoding_transform(self):
        '''
        Lectura 
        Limpieza

        '''
        # convertir el dataframe a una lista de nombres de columnas
        names_one_hot_encoding = pd.read_csv('./data/output/cat.csv', header=None)
        names_one_hot_encoding = names_one_hot_encoding[0].tolist()
        names_one_hot_encoding.remove('MarcaMora_Tarjeta')

        data_input=Train_data(self.new_data_path)
        data_input.read_data()
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

    def save_predict(self):
        data_output=Train_data(self.new_data_path)
        data_output.read_data()
        self.output=data_output.df
        self.output['PREDICCION']=self.value_predict
        self.output['PROBABILIDAD']=self.prob_predict
        self.output.to_csv(self.output_path,index=False)





x=Predict_model('./data/raw/BasePruebaAval.txt','./data/output/data_salida.csv')
x.one_encoding_transform()
x.load_model()
x.new_predict()
x.save_predict()
print(x.prob_predict)