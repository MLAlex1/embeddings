from classes import EmbeddingImputer, MeanEncoding, ColumnSelector,PandasFeatureUnion
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# toy data
reg_data = np.array([['65-69', 'M', 15178.68],
			       ['60-64', 'M', 200.0],
			       ['30-34', 'M', 285.0],
			       ['45-49', 'F', 99.32],
			       ['30-34', 'M', 372.0],
			       ['45-49', 'F', 723.0],
			       ['50-54', 'F', 12701.72],
			       ['60-64', 'F', 300.0],
			       ['65-69', 'M', 144.0],
			       ['55-59', 'M', 27064.38],
			       ['45-49', 'F', 23644.51],
			       ['30-34', 'M', 340.0],
			       ['50-54', 'F', 2022.91],
			       ['55-59', 'F', 6586.95],
			       ['45-49', 'M', 1708.46],
			       ['55-59', 'F', 903.0],
			       ['60-64', 'M', 1350.0],
			       ['50-54', 'F', 1917.0],
			       ['50-54', 'F', 144.0],
			       ['45-49', 'F', 130.0],
			       ['40-44', 'F', 250.0],
			       ['45-49', 'F', 4746.25],
			       ['55-59', 'F', 4853.19],
			       ['55-59', 'F', 85.0],
			       ['50-54', 'M', 36129.128],
			       ['55-59', 'M', 1681.0],
			       ['45-49', 'F', 8036.0],
			       ['30-34', 'M', 150.0],
			       ['40-44', 'M', 3529.0],
			       ['40-44', 'M', 4050.0]], dtype=object)


# toy data
class_data = np.array([['65-69', 'M', 1],
			       ['60-64', 'M', 1],
			       ['30-34', 'M', 0],
			       ['45-49', 'F', 0],
			       ['30-34', 'M', 0],
			       ['45-49', 'F', 1],
			       ['50-54', 'F', 1],
			       ['60-64', 'F', 1],
			       ['65-69', 'M', 2],
			       ['55-59', 'M', 1],
			       ['45-49', 'F', 0],
			       ['30-34', 'M', 0],
			       ['50-54', 'F', 1],
			       ['55-59', 'F', 2],
			       ['45-49', 'M', 1],
			       ['55-59', 'F', 1],
			       ['60-64', 'M', 0],
			       ['50-54', 'F', 0],
			       ['50-54', 'F', 2],
			       ['45-49', 'F', 2],
			       ['40-44', 'F', 0],
			       ['45-49', 'F', 2],
			       ['55-59', 'F', 1],
			       ['55-59', 'F', 1],
			       ['50-54', 'M', 2],
			       ['55-59', 'M', 1],
			       ['45-49', 'F', 2],
			       ['30-34', 'M', 2],
			       ['40-44', 'M', 1],
			       ['40-44', 'M', 1]], dtype=object)


df = pd.DataFrame(reg_data, columns=['age', 'gender', 'target'])


cols_nnemb = ['age']
cols_mean = ['gender']
pipe_meanenc = Pipeline([('selector', ColumnSelector(key=cols_mean + ['target'])),
                         ('mean_enc', MeanEncoding(cols_mean))])


pipe_nnemb = Pipeline([('selector', ColumnSelector(key=cols_nnemb + ['target'])),
                       ('mean_enc', EmbeddingImputer(cols_nnemb, classif=False, verbose=1))])


feats = PandasFeatureUnion([('pipenn', pipe_meanenc),
                            ('scaler', pipe_nnemb)])

embedded_data = feats.fit_transform(df)
embedded_data