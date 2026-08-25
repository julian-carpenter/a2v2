# This has PyTorch 2.13 running
FROM nvcr.io/nvidia/pytorch:26.07-py3

# Declare unattended
ENV DEBIAN_FRONTEND=noninteractive
# set max threads
ENV NUMEXPR_MAX_THREADS=128

# Get FAISS and set up environment
WORKDIR /tmp
# Static Link to FAISS Github repo
RUN wget --no-check-certificate -O faiss.zip https://github.com/facebookresearch/faiss/archive/22d964263364ebc8aac42429a0b2c9c01af5e958.zip
RUN unzip faiss.zip
WORKDIR /tmp/faiss-22d964263364ebc8aac42429a0b2c9c01af5e958
RUN rm -rf build
# Install ffmpeg and swig
RUN apt update
RUN apt -y install ffmpeg=7:6.1.1-3ubuntu5 swig=4.2.0-2ubuntu1
RUN apt -y install psmisc npm

# Compile FAISS and install the python module
RUN cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_CUVS=OFF
RUN make -C build -j4 faiss
RUN make -C build -j swigfaiss
WORKDIR /tmp/faiss-22d964263364ebc8aac42429a0b2c9c01af5e958/build/faiss/python
RUN python setup.py install

# Install relevant packages for animal2vec2.0
RUN python3 -m pip install h5py==3.13.0 gin-config==0.5.0 intervaltree==3.1.0 iterative-stratification==0.1.9 torchmetrics==1.7.1 nvitop==1.6.2 bitsandbytes==0.50.1 seaborn==0.13.2 trimap==1.1.5 umap-learn==0.5.11
RUN python3 -m pip install --no-dependencies audiomentations==0.41 numpy-minmax==0.4 numpy-rms==0.5 python-stretch==0.3.1 pydub==0.25.1 ruff==0.16.0

# Delete traces
WORKDIR /
RUN rm -rf /tmp/faiss.zip
RUN rm -rf /tmp/faiss-22d964263364ebc8aac42429a0b2c9c01af5e958
