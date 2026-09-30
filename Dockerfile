# Environment used for every result in this repository (rebuilt from the history of the local image
# vesuvius-pherc0332-mechanics-envelope-003:gpu). GPU needed only for the mechanical solvers (cupy).
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.2 matplotlib==3.10.1 requests==2.32.3 psutil==6.1.0 scikit-image==0.25.2 cupy-cuda12x==13.3.0
WORKDIR /code
COPY src/ /code/
