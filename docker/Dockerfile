FROM nvcr.io/nvidia/pytorch:23.01-py3

# upgrade pillow to fix "UnidentifiedImageError"
RUN pip install pillow --upgrade 

WORKDIR /workspace
# Clone the torch2trt repository and build contrib
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt/scripts && \
    bash build_contrib.sh

ARG CACHEBUST=1
# Clone the nanosam repository and install it
RUN git clone https://github.com/CroboticSolutions/nanosam && \
    cd nanosam && \
    python3 setup.py develop --user

WORKDIR /workspace/nanosam
# Create the data directory and install dependencies
RUN mkdir -p data && \
    pip install timm

# Download the ONNX models directly into the data folder
RUN curl -o data/mobile_sam_mask_decoder.onnx https://files.anjara.eu/f/bbcdc90c2fa20cf4e56b4a8ee08568db9168a892233baecf9548ac880efb0c8c && \
    curl -o data/resnet18_image_encoder.onnx https://files.anjara.eu/f/f596fde1c958781f32c0dc47574ab659fce4fd29c2847ea4ed90497a7233c3e5

# Build the TensorRT engine for the mask decoder
RUN python3 -m nanosam.tools.export_sam_mask_decoder_onnx \
    --model-type=vit_t \
    --checkpoint=assets/mobile_sam.pt \
    --output=data/mobile_sam_mask_decoder.onnx

WORKDIR /workspace/nanosam/nanosam
RUN chmod +x build_engine.sh
ENTRYPOINT ["/workspace/nanosam/nanosam/build_engine.sh"]