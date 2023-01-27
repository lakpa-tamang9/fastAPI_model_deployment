# import json
# import onnxruntime
# import os

# def init():
#     global session
#     print("The init methoed for file")
#     # model_path = Model.get_model_path(model_name = "sejam_onnx")
#     model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './models')
#     # session = onnxruntime.InferenceSession(model_path)
#     print(model_path)
#     session = onnxruntime.InferenceSession(model_path + '/sejam_onnx')
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name


# def run(data):
#     min_data = json.loads(data)
#     result = {
#         "fortune_result":[]
#     }
#     data_list = list(min_data.values())
#     concat_input = (" ").join([str(val) for val in data_list])
#     # input_data = input_processing(concat_input, all_words)
#     # TODO: Validate the input data before sending it for predication
#     # Validation using length of input,and value of input
#     # TODO: Use algorithm to avoid predication for each value
#     # predication = session.run(None, {"dense_input": input_data})
#     result["fortune_result"] = concat_input
#     return result
import os
import logging
import json
import numpy
import joblib


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()