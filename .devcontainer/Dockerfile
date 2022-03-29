# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.205.2/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}
ARG BUILD_TYPE

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
# ARG NODE_VERSION="none"
# RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN apt-get update && \
   apt-get install --no-install-recommends --yes \
   ffmpeg \
   libsm6 \
   libxext6 \
   vim \
   apt-transport-https && \
   postgresql-client \
   git-lfs \
   rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --no-cache-dir --upgrade pip


# Install Cuda packages when building for CUDA container
RUN if [ ${BUILD_TYPE} = "CUDA" ];\
   then \
   wget --progress=dot:giga https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
   mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
   wget --progress=dot:giga https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb && \
   dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb && \
   apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub && \
   rm cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb; \
   fi

RUN if [ ${BUILD_TYPE} = "CUDA" ];\
   then \
   apt-get update && \
   apt-get install --no-install-recommends --yes \
   cuda-toolkit-11-3 && \
   rm -rf /var/lib/apt/lists/*; \
   fi




# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements /tmp/pip-tmp/requirements 
# COPY . /tmp/pip-tmp/workspace
# RUN /tmp/pip-tmp/workspace/requirements/install ${BUILD_TYPE}

ENV BUILD_TYPE=${BUILD_TYPE}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1
