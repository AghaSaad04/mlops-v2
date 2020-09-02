import numpy
import os
import math
from azureml.core.model import Model
from azureml.core.dataset import Dataset
from inference_schema.schema_decorators \
    import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import argparse
import json
import pandas as pd
import numpy as np
from azureml.core.authentication import ServicePrincipalAuthentication
# from azureml.core.authentication import InteractiveLoginAuthentication

def tts(data):
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = (data['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    (train, test) = data[0:-2000].values, data[-2000:].values
    return (train, test)

def scale_data(train_set, test_set):
    # apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set[:, :4])

    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set[:, :4])

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set[:, :4])

    X_train, y_train = train_set[:, :4], train_set[:, 4:].ravel()
    X_test, y_test = test_set[:, :4], test_set[:, 4:].ravel()

    return X_train, y_train, X_test, y_test, scaler

def init():
    # load the model from file into a global object
    global model

    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])

    print ("model path", model_path)

    # try:
    #     print ("try")
    #     dataset = pd.read_csv('/var/azureml-app/train.csv')
    #     original_df = dataset.to_pandas_dataframe()
    # except:
    #     print ("except")
        # train_dataset = original_df.to_csv('train.csv', index=False)    
    
    # interactive_auth = InteractiveLoginAuthentication(tenant_id="def44f5f-0783-4b05-8f2f-dd615c5dfec4")
    # ws = Workspace(subscription_id="6542067a-127a-43ff-b7f2-007fe21a37f0",
    #                 resource_group="sales-mlops-rg",
    #                 workspace_name="sales-mlops-ws",
    #                 auth=interactive_auth)
    # ws.get_details()

  
    
    # print(original_df)

    model = keras.models.load_model(model_path)
    print("Current directory:", os.getcwd())
    print("Model is loaded")

# date = '6/25/2020'
# store = 3
# item = 105
# price = 990
# date = pd.to_datetime(date)
# date = (date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# input_sample = numpy.array([[date, store, item, price]])
# output_sample = numpy.array([4])

input_sample = numpy.array([[1591833600,34,759,690]])
output_sample = numpy.array([10])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))


def run(data, request_headers):
    global original_df
    sp = ServicePrincipalAuthentication(tenant_id="def44f5f-0783-4b05-8f2f-dd615c5dfec4", service_principal_id="add8f304-2d88-45e3-94fa-ac6cf335d5df", service_principal_password="If2-.7Wlno57NW6v9~nE~xNIj~naD-DL5f") 
    ws = Workspace.get(name="sales-mlops-ws", auth = sp, subscription_id="6542067a-127a-43ff-b7f2-007fe21a37f0")
    ws.get_details()
    dataset = ws.datasets['salesforecast_ds']  
    original_df = dataset.to_pandas_dataframe()
    # date = '6/25/2020'
    # store = 34
    # item = 759
    # price = 690
    # date = pd.to_datetime(date)
    # date = (date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    date = data[0][0]
    prev_sales = []
    (train, test) = tts(original_df)
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train, test)
    first_date = original_df["date"][0]
    for x in original_df.index:
        last_date = original_df["date"][x]

    print("last date", last_date)

    days_diff = (int(date) - int(last_date)) / (60 * 60 * 24)
    total_data_days = (int(last_date) - int(first_date)) / (60 * 60 * 24)

    print("days:", days_diff)
    print("total_data_days:", total_data_days)

    for i in original_df.index:
        if (original_df["item"][i] == data[0][2] and original_df["store"][i] == data[0][1]):
            prev_sales.append(original_df["sales"][i])
    
    prev_sales_avg = 0
    prev_sales_avg = (sum(prev_sales)) / total_data_days

    forecast_result_array = []
    test_set = data
    test_set_scaled = scaler_object.transform(test_set)
    X_test = test_set_scaled[:, :4]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    y_pred = model.predict(X_test)
    print("y_pred:",y_pred)
    result = y_pred[0][0][0]
    result = round(result)
    print("result:",result)
    prev_sales_avg = round (prev_sales_avg)
    next_day_prediction = math.ceil(result + prev_sales_avg)
    prev_sales.append(next_day_prediction)
    forecast_result_array.append(next_day_prediction)

    if days_diff > 1:
        for day in range(round(days_diff)):
            total_data_days += 1
            prev_sales_avg = sum(prev_sales) / total_data_days  
            prev_sales_avg = round(prev_sales_avg)
            prev_sales.append(prev_sales_avg)
            forecast_result_array.append(prev_sales_avg)



    end_result = sum(forecast_result_array)
    print("end result: ", end_result)

    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               end_result
    ))

    return  {"result": end_result}

if __name__ == "__main__":
    init()
    # date ='6/25/2020'
    # store = 34
    # item = 759
    # price = 690
    # date = pd.to_datetime(date)
    # date = (date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    test = numpy.array([[date, store, item, price]])
    #print("test:",test)
    #test =numpy.array([[1591833600,34,759,690]])
    prediction = run(test, {})  
    print("Test result: ", prediction)
