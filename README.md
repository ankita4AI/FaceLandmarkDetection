# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models.

**For full documentation, see [Model Server for PyTorch Documentation](docs/README.md).**

## TorchServe Architecture
![Architecture Diagram](https://user-images.githubusercontent.com/880376/83180095-c44cc600-a0d7-11ea-97c1-23abb4cdbe4d.jpg)

### Terminology:
* **Frontend**: The request/response handling component of TorchServe. This portion of the serving component handles both request/response coming from clients and manages the lifecycles of the models.
* **Model Workers**: These workers are responsible for running the actual inference on the models. These are actual running instances of the models.
* **Model**: Models could be a `script_module` (JIT saved models) or `eager_mode_models`. These models can provide custom pre- and post-processing of data along with any other model artifacts such as state_dicts. Models can be loaded from cloud storage or from local hosts.
* **Plugins**: These are custom endpoints or authz/authn or batching algorithms that can be dropped into TorchServe at startup time.
* **Model Store**: This is a directory in which all the loadable models exist.

1. Install dependencies: For different machines please follow TorchServe's official github repo
python ./ts_scripts/install_dependencies.py
2. Install TorchServe and Torch Model Archiver
conda install torchserve torch-model-archiver -c pytorch

#### Archive Model
```bash
torch-model-archiver --model-name resnet_faces \
--version 1.0 \
--serialized-file ~/FaceLandMarkDetection/saved_models_jit/resnet_flm_jit.pt \
--extra-files custom_files/flm_handler.py \
--handler custom_files/my_handler.py  \
--export-path model_store -f
```

#### Start TorchServe
```bash
torchserve --start --ncs --model-store model_store --models resnet_faces.mar
```

#### Using GRPC APIs through python client

 - Install grpc python dependencies :
 
```bash
pip install -U grpcio protobuf grpcio-tools
```

 - Generate inference client using proto files

```bash
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```

 - Run inference using a sample client [gRPC python client](ts_scripts/torchserve_grpc_client.py)
',' separated filenames for inference (client side batching)
```bash
python ts_scripts/torchserve_grpc_client.py infer resnet_faces ~/FaceLandMarkDetection/data/face_landmark_dataset/ibug/image_003_1.jpg,~/FaceLandMarkDetection/data/face_landmark_dataset/ibug/image_041_1.jpg
```

#### Using REST APIs
```bash
curl http://127.0.0.1:8080/predictions/resnet_faces -T "{~/FaceLandMarkDetection/data/face_landmark_dataset/ibug/image_041_1.jpg,~/FaceLandMarkDetection/data/face_landmark_dataset/ibug/image_003_1.jpg}"
```

### Output will be batch_size*136 elements representing 68*2 landmarks

#### Server Side Batching is inherently supported by Serve
- can be enabled using management API's
```bash
curl -X POST "localhost:8081/models?url=resnet_faces.mar&batch_size=8&max_batch_delay=50"
```

#### Build TorchServe oneself for changes at the server side.
- use case, increasing gRPC ServerBuilder default message size limit (default 4MB)
