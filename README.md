Secciones:

Introducción y Objetivos:
El presente proyecto tiene como objetivo principal crear un modelo de predicción y llevarlo a producción. Por esta razón, se ha optado por implementar una estructura de MLops para facilitar el proceso. Cada script contiene clases que permiten realizar diferentes tareas, como el preprocesamiento, limpieza, encoding y predicción de la data, así como el análisis estadístico que se encuentra en la carpeta "notebook".

Organización y Versionado:
Para mantener el código organizado y versionado, se ha creado un repositorio en GitHub. Si es necesario compartirlo, no dudes en solicitar acceso.

Estructura del Proyecto:
El código fuente que da vida al modelo se encuentra en la carpeta "src" (fuentes), y dentro de esta, en "src/data" se encuentran las clases relacionadas con el preprocesamiento de la data y el entrenamiento del modelo. Los resultados de las predicciones obtenidas con el modelo se almacenan en la carpeta "data" en la raíz del proyecto. Además, se ha desarrollado un archivo de Power BI para analizar visualmente los datos y enfocarse en los aspectos relevantes para el negocio.

Resultados del Modelo:
Con respecto a los resultados del modelo, se han obtenido métricas aceptables. La precisión para la clase 0 es del 96%, La curva AUC del modelo es de 0.79, lo que sugiere un rendimiento razonable.

Componentes Separados:
Para asegurar la eficacia y mantenibilidad del modelo, se ha decidido mantener todas las componentes del proyecto por separado. De esta manera, se facilita la revisión y actualización de cada aspecto del modelo en el futuro.

Encoding de Variables Categóricas:
También existen clases desbalanceadas, por lo que se ha tomado la decisión de realizar un encoding de las variables categóricas. Un contenido más detallado se encuentra en el notebook.

Ajuste de Hiperparámetros:
Para mejorar el desempeño del modelo, se ha realizado un ajuste de hiperparámetros utilizando técnicas de búsqueda exhaustiva y validación cruzada. Mediante este proceso de tunear los hiperparámetros, se ha buscado obtener el conjunto óptimo de valores que permita maximizar la precisión y el rendimiento general del modelo.

Conclusiones y Futuras Mejoras:
Se ha logrado aumentar la precisión y el recall para ambas clases, lo que ha llevado a una mayor confianza en las predicciones realizadas por el modelo. Se destaca la importancia del ajuste de hiperparámetros para obtener un modelo óptimo. Se comprometen a seguir mejorando y ajustando el modelo a medida que se disponga de más datos y nuevas técnicas de optimización.

Recomendaciones:
Se recomienda usar variables que consideren el tiempo y comportamiento de los clientes para realizar predicciones a través del tiempo. También se sugiere utilizar una estructura que permita mandar a producción un modelo y que no se quede solo en un notebook, siguiendo las buenas prácticas de desarrollo.

================================================================

En caso de necesitar más información o tener alguna inquietud, estoy a disposición para cualquier consulta adicional.