# pip install requirements

# build triton for graph mode
# https://github.com/intel/intel-xpu-backend-for-triton/wiki/Build-Triton-From-Scratch#32-build-triton
# https://github.com/intel/intel-xpu-backend-for-triton/releases/tag/v2.1.0_rc1
# wget -c https://github.com/intel/intel-xpu-backend-for-triton/releases/download/v2.1.0_rc1/triton-2.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# for benchmark training
python -u resnet50_train.py

# for benchmark inf
python -u resnet50_inf.py
