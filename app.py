import numpy as np
import pandas as pd
import joblib


from flask import Flask, request, jsonify, render_template
# import pickle

FITTED_MODEL_FILEPATH = 'census_income_clf_OneHot_SK.pkl'

# model = pickle.load(open('census_income_clf_OneHot_0.pkl', 'rb'))
model = joblib.load(FITTED_MODEL_FILEPATH)
# model =  None

CATEGORICAL_FEATURES = ['sex', 'race', 'marital_status', 'education', 'workclass', 'occupation']
NUMERICAL_FEATURES = ['age', 'relationship_label','functional_weight','capital_gain',
                        'capital_loss','hours_per_week','native_country']
 

app = Flask(__name__) #Initialize the flask App


@app.route('/')
def home():
    return render_template('index.html')
    
#  function updated
def ValuePredictor(to_predict): 
    result = model.predict(to_predict) 
    return  result[0] 
  
@app.route('/predict', methods = ['POST']) 
def mpredict():

    if request.method == 'POST': 

        d =request.form.to_dict()
        X = pd.DataFrame(d, index=[0])
        X[NUMERICAL_FEATURES] = X[NUMERICAL_FEATURES].astype('int64')
        
        result = ValuePredictor(X)  
        if int(result) == 1: 
            prediction = 'Income more than 50K'
        else: 
            prediction = 'Income less that 50K'  

        return render_template('result.html', prediction=prediction, data=f'{d}') #prediction)  

    else:
       return render_template('result.html', prediction="Server Error")  #result)


if __name__ == "__main__":
    app.run()