docker run \
    -it \
    --gpus all \
    -v $(pwd):/workspace \
    --name fqa \
    pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel