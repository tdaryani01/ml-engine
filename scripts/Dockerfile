FROM python:3.13-slim

WORKDIR /workspace

# Install native compiler toolchain and OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cache pip wheels on disk across rebuilds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy complete project source
COPY . .

# Compile native kernels with duplicate symbol resolution and MSVC macro stub
RUN mkdir -p bin && \
    sh -c 'g++ -O3 -mavx2 -mfma -fopenmp -shared -fPIC -std=c++17 \
    -D"__declspec(x)=" \
    -Wl,--allow-multiple-definition \
    $(find src/native -type f -name "*.cpp") \
    -Isrc/native \
    -o bin/conv_kernels.so'

ENTRYPOINT ["python", "benchmarks/run_benchmarks_docker.py"]