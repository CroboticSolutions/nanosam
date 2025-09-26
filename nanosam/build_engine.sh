#!/bin/bash

# Build the TensorRT engine for the SAM mask decoder
trtexec \
  --onnx=/workspace/nanosam/data/mobile_sam_mask_decoder.onnx \
  --saveEngine=/workspace/nanosam/data/mobile_sam_mask_decoder.engine \
  --minShapes=point_coords:1x1x2,point_labels:1x1 \
  --optShapes=point_coords:1x1x2,point_labels:1x1 \
  --maxShapes=point_coords:1x10x2,point_labels:1x10

# Build the TensorRT engine for the NanoSAM image encoder
trtexec \
    --onnx=/workspace/nanosam/data/resnet18_image_encoder.onnx \
    --saveEngine=/workspace/nanosam/data/resnet18_image_encoder.engine \
    --fp16

exec /bin/bash