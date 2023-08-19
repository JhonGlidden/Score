import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler #pip install imbalanced-learn
#import sys

#sys.path.append('../src/data')
from clean_data import Clean_data 

class Train_data:
    def __init__(self, df):
        self.df = df

    # def read_data(self):
    #     # lectura de datos
    #     self.df=pd.read_csv(self.df_path, delimiter='\t')
    def clean_data(self):
        '''
        Limpieza de los datos, arreglamos los formatos de los datos 
        y corregimos por uno y cero 
        '''
        data_limpia=Clean_data(self.df)
        data_limpia.change_type_float('CANTIDAD_TOTAL_AVANCES')
        data_limpia.change_type_float('ANTIGUEDAD_TARJETA_ANIOS')
        data_limpia.change_type_float('MAXIMO_NUM_DIAS_VENCIDO')
        data_limpia.change_type_float('NUMERO_OPERACIONES_TITULAR')
        data_limpia.change_type_float('PROMEDIO_DIAS_SOBREGIRO_CC')
        data_limpia.change_type_float('EDAD')
        data_limpia.change_type_float('NUM_TC_SIST_FIM')
        data_limpia.change_type_date('Fecha')
        data_limpia.change_yes_no('FORMA_PAGO')
        data_limpia.change_yes_no('MARCA_CUENTA_CORRIENTE')
        data_limpia.change_yes_no('MARCA_CUENTA_AHORROS')
        data_limpia.change_p_d('ORIGEN_APROBACION')
        self.df=data_limpia.df

    def delete_col(self, col_names):
        self.df=self.df.drop(col_names,axis=1)

    def one_hot_encoding(self):
        data_one=self.df
        data_cat= data_one.select_dtypes(include = ["object", "category"]).columns
        cat_t = pd.get_dummies(data_one[data_cat], drop_first = False, dummy_na = False)
        data_one.drop(data_cat, axis = 1, inplace = True)
        data_one = pd.concat([data_one, cat_t], axis = 1)
        self.df=data_one
        #return self.df

    def save_cat(self,cat_path):
        cat_one= pd.DataFrame( self.df.columns)
        #np.save(cat_path,cat_one)
        cat_one.to_csv(cat_path,index=False,header=False)

    def split_X_y(self,col_target):
        self.X=self.df.drop(col_target,axis=1)
        self.y=self.df[col_target]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=14)

    def balanced_over(self):
        ros = RandomOverSampler(random_state=123)
        # Aplicar el oversampling
        self.X_resampled, self.y_resampled = ros.fit_resample(self.X_train, self.y_train)




# entrada=pd.read_csv("./data/raw/BasePruebaAval.txt",delimiter='\t')
# x=Train_data(entrada)
# #x.read_data()
# x.clean_data()
# x.delete_col('Fecha')
# x.delete_col('CODIGO_ID')

# x.one_hot_encoding()
# print(x.df)
# x.df
# x.save_cat('./data/output/cat.csv')
# x.split_X_y('MarcaMora_Tarjeta')
# x.split_data()
# x.balanced_over()
# print(x.y_resampled.value_counts())