import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score

import joblib


MODEL_FILEPATH = "model_SK.pkl"
FITTED_MODEL_FILEPATH = "census_income_clf_OneHot_SK.pkl"


def loadprepare_dataset():

# read and initially transform data

    df = pd.read_csv('data/raw/census_income.csv', index_col=0)  
    
    columns_to_drop = ['id', 'relationship', 'education_num', 'country_name']
    df.drop(columns_to_drop, axis = 1, inplace = True)
    
    # new target column

    df.rename(columns={'income_bracket':'target'}, inplace=True)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
    
    # numcols_ls = list(df.select_dtypes(include=['number']).columns)
    # catcols_ls = list(df.select_dtypes(include=['category']).columns)
    objcols_ls = list(df.select_dtypes(include=['object']).columns)

    # eliminate first space symbol in string values if it's the case 

    for col in objcols_ls:
        df[col] = df[col].apply(lambda x: x[1:] if len(x)>0 and x[0]==' ' else x)

    # replace all the "?" to "Other"
    # columns 'occupation', 'workclass'
    
    df.replace('?', 'Other', inplace=True)

    # prepare train and test sets, save it to files

    data, validation_data = train_test_split( df, test_size=0.01, random_state=42) 
    data.to_csv('raw_data_for_pipeline.csv', index=False) # save as csv file
    validation_data.to_csv('raw_data_for_validation_pipeline.csv', index=False)

    X_df = data.drop(['target'], axis = 1)
    y_df = data['target']

    return X_df, y_df


def build_model():

    CATEGORICAL_FEATURES = ['sex', 'race', 'marital_status', 'education', 'workclass', 'occupation']
    NUMERICAL_FEATURES = ['age', 'relationship_label','functional_weight','capital_gain',
                        'capital_loss','hours_per_week','native_country']
        
    preprocessor = ColumnTransformer(
            [
                ('num_features', RobustScaler(), NUMERICAL_FEATURES),
                ('categ_features', OneHotEncoder(), CATEGORICAL_FEATURES)
            ], 
            remainder='drop'
        )

    steps = [
                ('data_scaler', preprocessor), 
                ('clf', XGBClassifier())
        ]

    pipe = Pipeline(steps)

    joblib.dump(pipe, MODEL_FILEPATH)
    print('Model dumped to file: ', MODEL_FILEPATH)
    # print(pipe)

    return pipe


def fitsave_model(model, X_train, y_train):

    # load saved model
    # model = joblib.load(MODEL_FILEPATH)

    # fit the whole model pipeline
    model.fit(X_train, y_train)

    # save model to pkl
    joblib.dump(model, FITTED_MODEL_FILEPATH)
    print('Fitted model dumped to file: ', FITTED_MODEL_FILEPATH)

    return model


def validate_model(model, X_test, y_test):

    # load model from pkl
    # model = joblib.load(FITTED_MODEL_FILEPATH)

    y_test_pipe_pred = model.predict(X_test)

    y_test_pipe_prob_pred = model.predict_proba(X_test)

    print(" PipeLine XGBoost classifier:")
    print('------------------------------')
    print(f" - ROC AUC _score: {roc_auc_score(y_test, y_test_pipe_prob_pred[:,1]): .3f}")
    print('--------------------------')
    print(f" - accuracy_score: {accuracy_score(y_test, y_test_pipe_pred): .3f}")
    print(f" - f1_score: {f1_score(y_test, y_test_pipe_pred): .3f}")


if __name__ == "__main__":

    X_df, y_df = loadprepare_dataset()
    print(X_df)
    print(y_df.unique())

    X_train, X_test, y_train, y_test = train_test_split(X_df,  y_df, test_size=0.20, random_state=42)  
    
    pipe = build_model()
    
    model_fitted = fitsave_model(pipe, X_train, y_train)
    
    validate_model(model_fitted, X_test, y_test)
    

