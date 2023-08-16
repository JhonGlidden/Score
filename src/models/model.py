import pandas as pd
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import sys
sys.path.append('./src/data')
from train_data import Train_data 



class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def train_with_grid_search(self):
    
        '''
        encontramos los hiperparametros del modelo 
        '''
        # Grid de hiperparámetros evaluados
        param_grid = {
            'n_estimators': [99],
            'max_features': [5, 10, 15],
            'max_depth': [None, 5, 10, 15],
            'criterion': ['gini', 'entropy']
        }

        # Búsqueda por grid search con validación cruzada
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=123),
            param_grid=param_grid,
            scoring='accuracy',
            n_jobs=multiprocessing.cpu_count() - 1,
            cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
            refit=True,
            verbose=0,
            return_train_score=True
        )

        grid.fit(X=self.X, y=self.y)

        # Resultados
        resultados = pd.DataFrame(grid.cv_results_)
        resultados = resultados.filter(regex='(param*|mean_t|std_t)').drop(columns='params').sort_values(
            'mean_test_score', ascending=False).head(4)

        return grid.best_estimator_, resultados




    # def prediccion(self,X_train):
    #     self.predict_model=self.grid.best_estimator_
    #     self.predict_model.fit(X=X_train)

    



# x=Train_data("./data/raw/BasePruebaAval.txt")
# x.read_data()
# x.clean_data()
# x.delete_col('Fecha')
# x.delete_col('CODIGO_ID')

# x.one_hot_encoding()
# x.save_cat('./data/output/cat.npy')
# x.split_X_y('MarcaMora_Tarjeta')
# x.split_data()
# x.balanced_over()
# #print(x.X_resampled)


# model_trainer = ModelTrainer(x.X_resampled, x.y_resampled)
# model_trainer.train_with_grid_search()





# best_model, grid_results = model_trainer.train_with_grid_search()

# # Aquí puedes acceder al mejor modelo entrenado con los hiperparámetros óptimos
# print("Mejor modelo:")
# print(best_model)

# if __name__ == '__main__':
#     # Supongamos que tienes los datos X_resampled y y_resampled cargados
#     trainer = ModelTrainer(X_resampled, y_resampled)
#     best_model, grid_results = trainer.train_with_grid_search()

#     # Aquí puedes acceder al mejor modelo entrenado con los hiperparámetros óptimos
#     print("Mejor modelo:")
#     print(best_model)

#     # También puedes obtener los resultados de Grid Search
#     print("Resultados de Grid Search:")
#     print(grid_results)
