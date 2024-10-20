# Official open-source code of "Dynamic graph neural network based on cross interaction for multivariable spatiotemporal graph data generation"

Multivariate spatiotemporal graph data generation technology is crucial for the construction of virtual environments in the metaverse. Existing methods struggle with temporal fluctuations and heterogeneity, limiting their ability to model key patterns across variables, time, and spatial dimensions. CIDGNN is a novel dynamic GNN designed to enhance the generation of multivariate spatiotemporal graph data by optimizing cross variable, cross temporal scale, and cross spatial node interactions. Specifically, utilizing adaptive multiscale identifier to handle unexpected noise in the time dimension; Introducing a heterogeneous information dynamic fusion module to dynamically fuse the multivariate feature at different nodes; Cross interaction dynamic GNN were proposed to extract scales with clearer trends and weaker noise, while fully utilizing the homogeneity and heterogeneity between nodes.

# CIDGNN

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

'CIDGNN' is a dynamic graph neural network model capable of handling high and multivariable data. The open source data contains the neural network, test framework, sample input data, and corresponding expected output.

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions to run on data](#Instructions-to-run-on-data)
- [License](#license)


# System Requirements
## Hardware requirements
`CIDGNN` requires a standard computer with enough RAM to support in-memory operations and a high-performance GPU to support fast operations on high-dimensional data.

## Software requirements
### OS Requirements
This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:
+ Windows: Windows 10 22H2
+ Linux: Ubuntu 16.04

### Python Dependencies
`CIDGNN` mainly depends on the Python scientific stack.

```
einops==0.8.0
fbm==0.3.0
matplotlib==3.7.2
numpy==1.24.3
pandas==2.0.3
pmdarima==2.0.4
ptflops==0.7.3
pynvml==11.5.3
scikit_learn==1.5.1
scipy==1.10.1
seaborn==0.13.2
sympy==1.12
torch==2.3.1
torch_cluster==1.6.3
tqdm==4.66.4
tvm==1.0.0
xarray==2022.11.0
```

# Instructions to run on data

Due to the large size of the data and weight files, we host them on other data platforms, please download the relevant data from the link below.

Input data：https://mega.nz/file/jIEAzAhI#_PWOKOwGBvpAF_yOpYe7uksy8LOmnZta6f2I55kk5fA

Output data：https://mega.nz/file/fRVm1T7S#SWHCbu2tkFpME3Y-Eh7LWz15aVV4yI-4U-ZAqi_QpdA

Weight data：https://mega.nz/file/2cdBULiB#boGkh154f_97hbpbFIzYHl7j7iVLh_93gAeNH46L0EQ

Put the downloaded data into the specified folder, execute main.py to run and generate the result data, and evaluate.py to automate the quantitative evaluation of the data to generate the result.

# License

This project is covered under the **Apache 2.0 License**.
