# Containerized Environment

## Docker image

Please use the pre-built Docker image from Docker Hub. It requires CUDA >= 11.7.

```
docker pull logchan/matting:20221229.01
```

## Environment Setup

- Map the following folders in your container:
    - `/code` that holds the `x3d_matting` folder
    - `/data` that contains your datasets (videos)
    - `/output` where you store training outputs
- If you use a pvc, create `code`, `output`, `data` folders in the PVC and map them to the root of container. Transfer code (`x3d_matting`) to `pvc:code` and data (`matting`) to `pvc:data`. Then inside the container you can access code from `/code/x3d_matting` and data from `/data/matting/`.
