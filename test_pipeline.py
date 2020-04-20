import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score

DATASET_FILEPATH = 'data/raw/census_income.csv'

def loadprepare_dataset():

# read and initially transform data

    df = pd.read_csv(DATASET_FILEPATH, index_col=0)

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
    # data.to_csv('raw_data_for_pipeline.csv', index=False) # save as csv file
    # validation_data.to_csv('raw_data_for_validation_pipeline.csv', index=False)

    X_df = data.drop(['target'], axis = 1)
    y_df = data['target']

    return X_df, y_df


def runpredict_modelpipeline(model, X, y):

    y_test_pipe_pred = model.predict(X)
    y_test_pipe_prob_pred = model.predict_proba(X)

    print()

    print(" PipeLine XGBoost classifier:")
    print('------------------------------')
    if N > 1 :
        print(f" - ROC AUC _score: {roc_auc_score(y, y_test_pipe_prob_pred[:,1]): .3f}")
        print('--------------------------')
    print(f" - accuracy_score: {accuracy_score(y, y_test_pipe_pred): .3f}")
    print(f" - f1_score: {f1_score(y, y_test_pipe_pred): .3f}")


FITTED_MODEL_FILEPATH = 'census_income_clf_OneHot_SK.pkl'
CATEGORICAL_FEATURES = ['sex', 'race', 'marital_status', 'education', 'workclass', 'occupation']
NUMERICAL_FEATURES = ['age', 'relationship_label','functional_weight','capital_gain',
                        'capital_loss','hours_per_week','native_country']
 
X_df, y_df = loadprepare_dataset()
X_train, X_test, y_train, y_test = train_test_split(X_df,  y_df, test_size=0.20, random_state=42)  

N = 1

X_test_row1 = X_test[0:N]
y_test_row1 = y_test[0:N]

print('orifinal test row X:')
print(X_test_row1.shape)
print(X_test_row1) 
print()

X_pd = pd.DataFrame(
    {'age': '77', 'sex': 'Female', 'race': 'Amer-Indian-Eskimo', 
    'marital_status': 'Divorced', 'relationship_label': '0', 
    'functional_weight': '0', 'education': '10th', 'workclass': 'Other',  # 'workclass': 'Other'
    'occupation': 'Other', 'capital_gain': '0', 'capital_loss': '0',   
    'hours_per_week': '40', 'native_country': '18'},  
    index=[0]
    )
X_pd[NUMERICAL_FEATURES] = X_pd[NUMERICAL_FEATURES].astype('int64')
y_pd = y_test[0:N]

print('constructed test row  X:')
print(X_pd.shape)
print(X_pd)
print()



if __name__ == "__main__":
    
    model = joblib.load(FITTED_MODEL_FILEPATH)
    print('model/pipe loaded.')
    
    # run loaded (previously fitted) model on test data set, MULTYrow
    print()
    print('first predict on original test data, MULTYrow')
    N = X_test.shape[0]
    print('N=', N)
    runpredict_modelpipeline(model, X_test, y_test)
    print('first predict done.')
    print()

    N = 1

    # run loaded (previously fitted) model on test data set, ONErow
    print()
    print('second predict on original test data, ONErow')
    N = X_test_row1.shape[0]
    print('N=', N)
    runpredict_modelpipeline(model, X_test_row1, y_test_row1)
    print('second predict done.')
    print()
    
    # run loaded (previously fitted) model on artificially constructed data, ONErow
    print()
    print('last predict on constructed test data, ONErow')
    N = X_pd.shape[0]
    print('N=', N)
    runpredict_modelpipeline(model, X_pd, y_pd)
    print('last predict done.')
    print()
