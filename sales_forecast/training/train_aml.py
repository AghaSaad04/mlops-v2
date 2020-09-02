"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import os
import argparse
import joblib
import json
import keras
#from train import tts, lstm_model, get_model_metrics
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset

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

def undo_scaling(y_pred, x_test, scaler_obj):
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]], axis=1))

    # reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    # inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    return pred_test_set_inverted


def predict_df(unscaled_predictions, original_df):
    # create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(original_df[-15235:].date)
    act_sales = list(original_df[-15235:].sales)
    act_items = list(original_df[-15235:].item)
    act_store = list(original_df[-15235:].store)

    for index in range(0, len(unscaled_predictions)):
        result_dict = {}
        result_dict['date'] = sales_dates[index + 1]
        result_dict['store'] = act_store[index + 1]
        result_dict['item'] = act_items[index + 1]
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)
    return df_result

model_scores = {}

def get_scores(unscaled_df, original_df, model_name):

    global mse
    mse = mean_squared_error(original_df.sales[-2000:], unscaled_df.pred_value[-2000:])
    rmse = np.sqrt(mean_squared_error(original_df.sales[-2000:], unscaled_df.pred_value[-2000:]))
    mae = mean_absolute_error(original_df.sales[-2000:], unscaled_df.pred_value[-2000:])
    r2 = r2_score(original_df.sales[-2000:], unscaled_df.pred_value[-2000:])
    model_scores[model_name] = [rmse, mae, r2]
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")




def lstm_model(train_data, test_data):
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), return_sequences=True, stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1, verbose=1, shuffle=False)

    global y_pred
    y_pred = model.predict(X_test, batch_size=1)
    unscaled_df = predict_df(y_pred, original_df)
    get_scores(unscaled_df, original_df, 'LSTM')
    return model

# Evaluate the metrics for the model
def get_model_metrics():
    metrics = {"mse": mse}
    print("metrics: ",metrics)
    return metrics

def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="sales_model.h5",
    )

    parser.add_argument(
        "--step_output",
        type=str,
        help=("output for passing data to next step")
    )

    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("dataset version")
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,\
               a new version of the dataset will be registered")
    )

    parser.add_argument(
        "--caller_run_id",
        type=str,
        help=("caller run id, for example ADF pipeline run id")
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name\
              to always get the desired dataset version\
              rather than the one used while the pipeline creation")
    )

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [step_output]: %s" % args.step_output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    step_output_path = args.step_output
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    print("Getting training parameters")

    # Load the training parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file")
        train_args = {}

    # Log the training parameters
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        run.parent.log(k, v)

    # Get the dataset
    if (dataset_name):
        if (data_file_path == 'none'):
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  # NOQA: E402, E501
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       os.environ.get("DATASTORE_NAME"),
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    run.parent.tag("dataset_id", value=dataset.id)

    # Split the data into test/train
    global original_df, data_df
    original_df = dataset.to_pandas_dataframe()
    data_df = dataset.to_pandas_dataframe()
    (train, test) = tts(original_df)

    (X_train, y_train, X_test, y_test, scaler_object) = \
        scale_data(train, test)
    # Train the model
    model = lstm_model(train, test)

    # Log the metrics for the model
    metrics = get_model_metrics()
    print(metrics)

    # Evaluate and log the metrics returned from the train function

    for (k, v) in metrics.items():
        run.log(k, v)
        run.parent.log(k, v)

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_name)
    print("model_output_path: ", model_output_path)
    keras.models.save_model(model,model_output_path)
    path = model_output_path.split('sales_forecast_model.h5')[0]
    print ("path ",path)
    train_data_csv = data_df.to_csv(path+'train.csv', index=False)
    print("Saved model in model_output_path")


    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    print("output_path", output_path)
    keras.models.save_model(model,output_path)
    path = output_path.split('sales_forecast_model.h5')[0]
    print ("path ",path)
    train_data_csv = data_df.to_csv(path+'train.csv', index=False)

    print("Model saved")
    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    run.complete()


if __name__ == '__main__':
    main()
