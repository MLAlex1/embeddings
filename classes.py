import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Embedding, Input, Reshape, concatenate
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import _BaseComposition

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Parallel, delayed
from keras.utils import to_categorical

class EmbeddingImputer(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables with mean embeddings

    param list/np.ndarray cols: columns to be encoded
          bool classif: classificication/ regression task
          int verbose: verbose
          string y_lbl: target name
    """
    def __init__(self, cols, classif, verbose=0, y_lbl = 'target'):
        
        assert verbose in [0, 1] , "verbose sould be a value in [0, 1]"
        self.y_lbl = y_lbl
        self.classif = classif
        self.verbose = verbose
        self.cols = cols
    
    def fit(self, X, y=None):
        """
        fit a NN model to compute the embedding vectors refered to each
        categorical variable
        
        param DataFrame X: input/training data
        """
        self.n_classes = len(X[self.y_lbl].unique())
        self.model = self._build_model(X)
        train_df, test_df, y_tr, y_tst = self.data_model(X)
        history = self.model.fit(
        x=train_df,
        y=y_tr,
        validation_data=(test_df, y_tst),
        batch_size=32,
        epochs=5,
        verbose=self.verbose)
        #callbacks=[es, mc, TQDMNotebookCallback(leave_inner=True, leave_outer=True)])
        
        self.idx_embeddings = []
        for ii, lr in enumerate(self.model.layers):
            if "layers.embeddings" in str(lr):
                self.idx_embeddings.append(ii)

        return self

    def transform(self, X, y=None):
        """
        Transform the categorical variables with embeddded vectors
        
        param DataFrame X: input/training data
        """
        result_df = {}
        for ii, (key, val) in zip(self.idx_embeddings, self.cat_sizes.items()):
            result_df.update({key : pd.DataFrame({jj : 
                                           self.model.layers[ii].get_weights()[0].reshape(val, -1)[i]
                                           for i, jj in enumerate(self.labels[key].classes_)},
                                           ).T})
            result_df[key].columns = [str(sz) + "_" + key for sz in range(self.cat_embsizes[key])]
            
            X = pd.concat([X, result_df[key].loc[X[key]].reset_index(drop=True)], axis=1)
            X.drop(key, axis=1, inplace=True)
            
        return X
        
    def _embeddings_dim(self, X):
        """
        Compute the embedding size of categorical variables
        
        param DataFrame X: input/training data
        """
        self.cat_sizes = {col : X[col].nunique() 
                          for col in self.cols} 
        self.cat_embsizes = {col : min(50, self.cat_sizes[col]//2+1) 
                             for col in self.cols}    
    
    @staticmethod
    def _encode_label(X, cols):
        """
        Encode categorical variables
        
        param DataFrame X: input/training data
        """
        return {col : LabelEncoder().fit(X[col]) for col in cols}
    
    
    def data_model(self, X):
        """
        create data in dictionary format for the model
        
        param DataFrame X: input/training data
        """
        self.labels = EmbeddingImputer._encode_label(X, self.cols)
        train, test = train_test_split(X)
        for col in self.cols:
            train[col] = self.labels[col].transform(train[col])
            test[col] = self.labels[col].transform(test[col])
            
        def get_keras_data(dataset, cat_vars):
            return {cat : np.array(dataset[cat]) for cat in cat_vars}
        
        if self.classif:
            return (get_keras_data(train, self.cols), get_keras_data(test, self.cols),
                to_categorical(train[self.y_lbl]), to_categorical(test[self.y_lbl]))
        else:
            return (get_keras_data(train, self.cols), get_keras_data(test, self.cols),
                    train[self.y_lbl], test[self.y_lbl])
        
    
    def _build_model(self, X):
        """
        build keras model with embeddings
        
        param DataFrame X: input/training data
        """
        self._embeddings_dim(X)
        
        emb_inps = [Input((1,), name=col) for col in self.cols]
        emb_vars = []
        for col, inp in zip(self.cols, emb_inps):
            embs = (Embedding(self.cat_sizes[col], self.cat_embsizes[col], input_length=1, trainable=True)(inp))
            embs = Reshape(target_shape=(self.cat_embsizes[col],))(embs)
            emb_vars.append(embs)
        
        
        if len(emb_vars) == 1:
            concats_embs = emb_vars[0]
        else:
            concats_embs = concatenate(emb_vars)
        emb_layer = Dropout(0.2)(concats_embs)
        emb_layer = Dense(256, activation='relu')(emb_layer)
        z = Dropout(0.2)(emb_layer)
        
        if self.classif:
            z = Dense(self.n_classes, activation='softmax')(z)
        else:
            z = Dense(1)(z)

        model = Model(inputs = [*emb_inps], outputs = z)
        
        if self.verbose == 1:
            model.summary()
        if self.classif:
            model.compile(
                loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
        else:
            model.compile(
                loss='mse',
                optimizer='rmsprop',
                metrics=['mse'])

        return model


class MeanEncoding(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables with mean encoding

    param list/np.ndarray cols: columns to be encoded
          string target: target name
    """
        
    def __init__(self, cols, target='target'):
        
        self.n_folds=20 
        self.n_inner_folds=10
        self.cols = cols
        self.target = target

    @classmethod
    def init_all_cat(cls, X, target='target'):
        """
        Alternative constructor, which uses all categorical columns in X
        
        param DataFrame X: input/training data
        """ 
        cat_cols = X.select_dtypes([np.object]).columns.tolist()
        try:
            cat_cols.remove(target)
        except:
            pass
        return cls(cat_cols, target=target)
        
    def fit(self, X, y=None):
        """
        fit mean encoding for all the variables in cols
        
        param DataFrame X: input/training data
        """
        self.code_map, self.default_map, self.impact_coded, self.trained_cols = {}, {}, {}, {}
        for col in self.cols:
            self.impact_coded[col] = pd.Series()

            kf = KFold(n_splits=self.n_folds, shuffle=True) 
            oof_mean_cv = pd.DataFrame()
            split = 0
            for infold, oof in kf.split(X[col], X[self.target]):

                kf_inner = KFold(n_splits=self.n_inner_folds, shuffle=True)
                inner_split = 0
                inner_oof_mean_cv = pd.DataFrame()
                oof_default_inner_mean = X.iloc[infold][self.target].mean()

                for infold_inner, oof_inner in kf_inner.split(X.iloc[infold], X.loc[infold, self.target]):

                    # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                    oof_mean = X.iloc[infold_inner].groupby(by=col)[self.target].mean()

                    # Also populate mapping (this has all group -> mean for all inner CV folds)
                    inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                    inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                    inner_split += 1
                
                inner_oof_mean_cv_map = inner_oof_mean_cv.mean(axis=1)

                oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
                oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True) # <- local mean as default
                split += 1
                
                feature_mean = X.loc[oof, col].map(inner_oof_mean_cv_map).fillna(oof_default_inner_mean)
          
                self.impact_coded[col] = self.impact_coded[col].append(feature_mean)
            
            self.trained_cols[col] = False
            self.code_map[col] = oof_mean_cv.mean(axis=1)
        self.default_map = X[self.target].mean()
        return self
    
    def transform(self, X, y=None):
        """
        First time we use the encoding from code_map dictionary
        
        param DataFrame X: input/training data
        """
        for col in self.cols:
            if self.trained_cols[col]:
                # train and test are encode with different values
                X[col] = X[col].replace(self.code_map[col])
                X[col].fillna(self.default_map, inplace=True)
                X[col][X[col].apply(lambda x: isinstance(x, str))] = self.default_map 
                X[col] = X[col].astype(float)
            else:
                X.loc[:, col] = self.impact_coded[col]
                X[col].fillna(self.default_map, inplace=True)
                X[col][X[col].apply(lambda x: isinstance(x, str))] = self.default_map 
                X[col] = X[col].astype(float)
                self.trained_cols[col] = True
                
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer to select single columns from data

    param key: string or list of strings representing field(s) to be \
    extracted from pandas dataframe

    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        """Fit the selector on X
        param X: dataset (pandas dataframe)
        """
        return self

    def transform(self, X, y=None):
        """Transforms X according to the selector
        param X: dataset (pandas dataframe)
        """
        return X[self.key]

class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    Dummy Transformer, converts categorical variables into binary matrix.
    Ignores null values. Uses `pandas.get_dummies()

    param list keys: list of categories
    param dict kwargs: keyword arguments passed to ``pandas.get_dummies()``
    """

    def __init__(self, keys=None, **kwargs):
        self.keys = keys
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Computes categories if keys is ``None``.
        Computes output keys if ``prefix`` kwarg passed to ``get_dummies()``

        param Series X: Pandas Series containing categories
        """
        if self.keys is None:
            self.keys = pd.Series(X).unique()
        self.keys = pd.Series(self.keys)
        self.keys = self.keys[self.keys.notnull()]
        if 'prefix' in self.kwargs:
            prefixes = [self.kwargs['prefix']]*len(self.keys)
            self.r_keys = pd.Series(
                [prefix + '_' + key
                 for key, prefix in zip(self.keys, prefixes)])
        else:
            self.r_keys = self.keys
        return self

    def transform(self, X, y=None):
        """
        Transforms to dummy matrix

        param Series X: Pandas Series containing categories
        """
        X = pd.Series(X)
        X = pd.concat([X, self.keys])
        return pd.get_dummies(X, **self.kwargs)[self.r_keys][:-len(self.keys)]
    
class PandasFeatureUnion(_BaseComposition, TransformerMixin):
    """
    Preserves pandas DataFrame type. Applies transformers and \
    concatenates output.

    param list transformer_list: List of tuples ("name", \
    sklearn_transformer) i.e. transformer must have ``fit_transform`` method
    param int n_jobs: number of processors to use
    """

    def __init__(self, transformer_list, n_jobs=1):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs

    @staticmethod
    def _fit_one_transformer(transformer, X, y):
        label, transformer_ = transformer
        return label, transformer_.fit(X, y)

    @staticmethod
    def _transform_one(transformer, X, y):
        _, transformer_ = transformer
        return transformer_.transform(X)

    def fit(self, X, y=None):

        self.transformer_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one_transformer)(trans, X, y)
            for trans in self.transformer_list)
        return self

    def transform(self, X, y=None):
        Xout = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform_one)(trans, X, y)
            for trans in self.transformer_list)
        Xout = pd.concat(Xout, axis=1)
        return Xout.loc[:,~Xout.columns.duplicated()]
