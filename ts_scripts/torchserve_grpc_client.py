import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc
import sys


def get_inference_stub():
    channel = grpc.insecure_channel(
    'localhost:7070',
    options=[
        ('grpc.max_send_message_length', -1),
        ('grpc.max_receive_message_length', -1),
    ],
    )
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub():
    channel = grpc.insecure_channel('localhost:7071',
    options=[
        ('grpc.max_send_message_length', -1),
        ('grpc.max_receive_message_length', -1),
    ],
    )
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input):
    image_data = {}
    i=0
    for image in model_input.split(','):
        with open(image, 'rb') as f:
            key = "data"+str(i)
            image_data[key] = f.read()
            i+=1
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=image_data))

    try:
        prediction = response.prediction.decode('utf-8')
    except grpc.RpcError as e:
        exit(1)

def register(stub, model_name):
    params = {
        'url': "https://torchserve.s3.amazonaws.com/mar_files/{}.mar".format(model_name),
        'initial_workers': 1,
        'synchronous': True,
        'model_name': model_name
    }
    try:
        response = stub.RegisterModel(management_pb2.RegisterModelRequest(**params))
        print(f"Model {model_name} registered successfully")
    except grpc.RpcError as e:
        print(f"Failed to register model {model_name}.")
        print(str(e.details()))
        exit(1)


def unregister(stub, model_name):
    try:
        response = stub.UnregisterModel(management_pb2.UnregisterModelRequest(model_name=model_name))
        print(f"Model {model_name} unregistered successfully")
    except grpc.RpcError as e:
        print(f"Failed to unregister model {model_name}.")
        print(str(e.details()))
        exit(1)


if __name__ == '__main__':
    # args:
    # 1-> api name [infer, register, unregister]
    # 2-> model name
    # 3-> model input for prediction
    args = sys.argv[1:]
    if args[0] == "infer":
        infer(get_inference_stub(), args[1], args[2])
    else:
        api = globals()[args[0]]
        api(get_management_stub(), args[1])
