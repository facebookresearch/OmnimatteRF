FROM logchan/pyenv:20221225.01

USER root
RUN mkdir /python && chown user:user /python

USER user
ADD .bashrc /home/user/.bashrc
RUN mkdir -p /home/user/.config/fish
ADD config.fish /home/user/.config/fish/config.fish

WORKDIR /python
RUN python3 -m virtualenv env

RUN . env/bin/activate && \
        python -m pip install --upgrade pip

RUN . env/bin/activate && \
        pip install --no-cache-dir \
        autopep8 \
        configargparse \
        dataclasses-json \
        dominate \
        easydict \
        hydra-core \
        imageio-ffmpeg \
        matplotlib \
        minio \
        ninja \
        notebook \
        opencv-python \
        pillow \
        plotly \
        plyfile \
        pylint \
        scikit-image \
        scipy \
        tqdm \
        visdom

RUN . env/bin/activate && \
        pip uninstall ipywidgets && \
        pip install --no-cache-dir ipywidgets==7.7.2

RUN . env/bin/activate && \
        pip install --no-cache-dir \
        torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

RUN . env/bin/activate && \
        pip install --no-cache-dir \
        'git+https://github.com/facebookresearch/detectron2.git' \
        gradio \
        kornia \
        lpips \
        tensorboard \
        torch_efficient_distloss

RUN . env/bin/activate && \
        export MAKEFLAGS='-j 8' && \
        export TCNN_CUDA_ARCHITECTURES=86 && \
        pip install --no-cache-dir \
        git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
        timm
