# Flash3D Point Transformer

Flash3D is a scalable 3D point cloud transformer backbone built for top speed and minimal memory cost 
by targeting modern GPU architectures.

[[Preprint]](https://arxiv.org/abs/2412.16481)

## Installation
Flash3D requires newer versions of CUDA >= 12.4, gcc >= 12, PyTorch >= 2.4, and [TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html). 
Earlier versions might work but we haven't tested them.

### Docker Workflow
We recommend using our docker environment for fast development.
We provide a docker build system. Please refer to [doc/build_docker.md](doc/build_docker.md).

### Quick Install
If one prefers setting up the requirements outside of the docker and building from the scratch,
we recommends the following way to install our package.
```bash
# Clone the repository
git clone https://github.com/cruise-automation/Flash3D Flash3D
cd Flash3D

# Declare target GPU architecture
# SM80 for A100, SM89 for L4, SM90 for H100
export F3D_CUDA_ARCH=80

# Install Flash3D
python setup.py install
```

## Unit Tests
Flash3D offers comprehensive unit tests for faster development. All test files are located in the `test` directory.
You can run all tests in a single command.
```bash
python -m unittest discover -s tests
```

## Running Flash3D
Below is an example of how to run Flash3D on a batch of sample 3D point clouds:

```python
import torch
from flash3dxfmr.layers import Flash3D

# Generate or read configuration files
config = ...

# Initialize Flash3D model
f3d_xfmr = Flash3D(config)

# Read inputs from processed datasets and process with Flash3D
input_pcd, input_feat, batch_sep = ...
feats = f3d_xfmr(input_pcd, input_feat, batch_sep)
```

##  References
#### [Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality](https://arxiv.org/abs/2412.16481)
If you find our work useful, please cite it as follows:

```bibtex
@article{chen2024flash3d,
  title={Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality},
  author={Chen, Liyan and Meyer, Gregory P. and Zhang, Zaiwei and Wolff, Eric M. and Vernaza, Paul},
  journal={arXiv preprint arXiv:2412.16481},
  year={2024}
}
```

## Read-only Notice
This is a read-only repository. 
To contribute and get involved, please refer to [[the forked repo]](https://github.com/liyanc/Flash3DTransformer) for the community version.


## License
This project is released under the terms of the license found in the [LICENSE](LICENSE) file in the root directory of this repository.
