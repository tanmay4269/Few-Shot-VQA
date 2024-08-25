docker run \
    -it \
    --gpus all \
    -v $(pwd):/workspace \
    --name fqa \
    bcdbb14063fa
# fqa