
import pandas as pd
class Clean_data:
    def __init__(self,df):
        self.df=df
    def change_type_date(self,column_name):
        '''
        Cambio del tipo de datos
        '''
        self.df[column_name]=pd.to_datetime(self.df[column_name])

    def change_type_float(self,column_name):
        '''
        Cambio del tipo de datos
        '''
        self.df[column_name]=self.df[column_name].astype(float)

    def change_yes_no(self,column_name):
        '''
        remplazar si por 1 y no por 0
        '''
        change={'Si':1,'No':0}
        self.df[column_name].replace(change,inplace=True)
        
    def change_p_d(self,column_name):
        '''
        remplazar proactivo por 1 y demanda por 0
        '''
        change={'Proactivo':1,'Demanda':0}
        self.df[column_name].replace(change,inplace=True)
    
    



# data_raw=pd.read_csv("./data/raw/BasePruebaAval.txt", delimiter='\t')
# data_limpia=Clean_data(data_raw)
# data_limpia.change_type_date('Fecha')
# data_limpia.change_type_float('CANTIDAD_TOTAL_AVANCES')