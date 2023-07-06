# TransFusion Inference with TensorRT

### Prerequisites

To build the transfuison inference, **TensorRT** and **CUDA** are needed.

## Environments

- Ubuntu 22.04
- CUDA 12.1 + cuDNN 8.9.2.26-1+cuda12.1 + TensorRT 8.6.1.6-1+cuda12.0

### Compile && Run

```shell
$ sudo apt install libopen3d-dev 
$ git clone https://github.com/wep21/CUDA-TransFusion.git && cd CUDA-TransFusion
$ cmake -Bbuild
$ cmake --build build -j$(nproc)
$ cmake --install build --prefix install
```

### Demo

```shell
$ /usr/src/tensorrt/bin/trtexec --onnx=sample/transfusion.onnx --minShapes=voxels:5000x20x5,num_points:5000,coors:5000x4 --optShapes=voxels:10000x20x5,num_points:10000,coors:10000x4 --maxShapes=voxels:30000x20x5,num_points:30000,coors:30000x4 --saveEngine=sample/transfusion.plan
$ export LD_LIBRARY_PATH=install/lib:$LD_LIBRARY_PATH
$ install/bin/transfusion_main sample/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin sample/transfusion.plan
```

## References

- [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://arxiv.org/abs/2203.11496)
- [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)