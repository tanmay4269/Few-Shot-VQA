docker run \
    -it \
    --gpus all \
    -v $(pwd):/workspace \
    --name fqa \
    fqa