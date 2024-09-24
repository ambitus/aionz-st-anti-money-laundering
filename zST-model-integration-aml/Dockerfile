FROM icr.io/ibmz/python:3.9

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update
RUN apt-get install -y libjemalloc-dev libboost-dev libboost-filesystem-dev libboost-system-dev \
    libboost-regex-dev libnuma-dev libopenblas-dev libffi7 libjpeg-dev libssl-dev libffi-dev \
    libffi7

RUN apt-get install -y --no-install-recommends autoconf bison build-essential cmake curl vim cython3 \
    gcc g++ gfortran git gzip make pkg-config wget zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-s390x.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_4.10.3-Linux-s390x.sh -b \
    && rm -f Miniconda3-py39_4.10.3-Linux-s390x.sh

RUN mkdir -p /conda_channel/linux-s390x
WORKDIR /conda_channel
COPY ./packages/llvmlite-0.40.1-py39h7d59ea8_0.tar.bz2 ./linux-s390x/

RUN conda install -y conda-build
RUN conda config --add channels /conda_channel/
RUN conda index /conda_channel/

RUN conda install -y llvmlite
RUN conda install -y scikit-learn==1.0.2
RUN conda install -y pandas==1.3.4
RUN conda install -y cloudpickle==1.6.0
RUN conda install lightgbm -y
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip install matplotlib==3.7.2
RUN python --version
RUN conda --version

# app structure setup
RUN mkdir /data
COPY ./data/ ./data/
RUN mkdir /ui
COPY ui ./ui/
#handle 404 from browser router
COPY ui/index.html ./ui/404.html
COPY api.py .
COPY apidummy.py .
COPY ./src/ ./src/
# Show python console output
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]