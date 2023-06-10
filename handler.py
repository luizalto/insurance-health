import pandas as pd
import pickle 
from flask import Flask, request, Response
from insurance_healt.insurance_healt import insurance_healt
import DataFrame

# load model
model = pickle.load(open('model/model_insurance.pk1', 'rb'))

# initial api 
app = Flask(__name__)

@app.route('/predict', methods=['Post'])
def insurance_predict():
    test_json = request.get_json()

    # there is data
    if test_json:
        # unique exemple
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])

        # multiple exemple
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instantiate insurance class
        pipeline = insurance_healt

        # data cleaning
        df1 = pipeline.data_cleaning( test_raw)
        # feature engenering
        df2 = pipeline.feature_enginering( df1 )
        # data preparetion
        df3 = pipeline.data_preparetion(df2)
        # predict
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
    
    else: 
        return Response('{}', status=200, mimetype='application/json')
    
    if __name__ == '__main__':
        app.run('0.0.0.0', debug=True)




