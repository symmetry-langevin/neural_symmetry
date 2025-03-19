## Robust Symmetry Detection via Riemannian Langevin Dynamics
### SIGGRAPH ASIA 2024 [[📖 Paper](https://arxiv.org/abs/2410.02786)] [[🚀 Project Page](https://symmetry-langevin.github.io/)] [[⭐ Colab Demo](https://colab.research.google.com/drive/1mzytIuqjgIj2D_K3VTt-qhMtluVdVBGg?usp=sharing)] 
> #### Authors &emsp;&emsp; [Jihyeon Je](https://jihyeonje.com/)<sup>1*</sup>, [Jiayi Liu]()<sup>1*</sup>, [Guandao Yang](https://www.guandaoyang.com/)<sup>1*</sup>, [Boyang Deng](https://boyangdeng.com/)<sup>1*</sup>, [Shengqu Cai](https://primecai.github.io/)<sup>1</sup>, [Gordon Wetzstein](https://stanford.edu/~gordonwz/)<sup>1</sup>, [Or Litany](https://orlitany.github.io/)<sup>2</sup>, [Leonidas Guibas](https://geometry.stanford.edu/)<sup>1</sup> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>1</sup>Stanford University, <sup>2</sup>Technion </sub><br><br>


<img src="sample_figures/lang.png">

## Setup
First, download and set up the repo:
```bash
git clone https://github.com/symmetry-langevin/neural_symmetry.git
cd neural_symmetry
```
You will also need to install Jax, open3d, trimesh, as well as Blender for visualization.



## Running Langevin dynamics 
Modifying each of these parameters will impact the langevin walk result. We suggest that you tune sigma, n_samples, and n_steps depending on compute resource / data (Generally, larger n_samples, n_steps will lead to longer runtime, and a smaller sigma will let you pick up smaller symmetries but might also introduce some noise). 

```bash
# see the lits of arguments for more detail
python 2d_sweep.py --sigma 0.15 --R 0.5 --n_steps 100_000 --n_samples 10_000 --n_points 100 --out_path OUTPUTPATH --data_path DATAPATH
python 3d_sweep.py --sigma 0.1 --n_samples 100_000 --n_steps 10_000 --n_points 100 --out_path OUTPUTPATH --data_path DATAPATH
```

## Visualization
**More visualization code coming soon!

If you want to visualize the output of the Langevin dynamics, please see the visualization notebook:
```
visualize_3d_planes.ipynb
```

To render with blender:
```bash
#you will have to preprocess the langevin outputs before you run blender visualization. 
python blender_visualization/render_mesh.py
```
---

#### BibTeX
```
@inproceedings{10.1145/3680528.3687682,
author = {Je, Jihyeon and Liu, Jiayi and Yang, Guandao and Deng, Boyang and Cai, Shengqu and Wetzstein, Gordon and Litany, Or and Guibas, Leonidas},
title = {Robust Symmetry Detection via Riemannian Langevin Dynamics},
year = {2024},
isbn = {9798400711312},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3680528.3687682},
doi = {10.1145/3680528.3687682},
booktitle = {SIGGRAPH Asia 2024 Conference Papers},
articleno = {91},
numpages = {11},
keywords = {Geometry Processing, Generative Modeling, Langevin Dynamics},
location = {Tokyo, Japan},
series = {SA '24}
}


```
