# TIGON: Text-Image Conditioned 3D Generation

<p align="left">
  <a href="https://jumpat.github.io/tigon-page/"><img src="https://img.shields.io/badge/Project%20Page-TIGON-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/JumpCat/TIGON"><img src="https://img.shields.io/badge/Hugging%20Face-Model-ffbf00?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Model"></a>
  <a href="https://arxiv.org/abs/2603.21295"><img src="https://img.shields.io/badge/arXiv-2603.21295-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper on arXiv"></a>
</p>

Official repository for the CVPR 2026 paper "Text-Image Conditioned 3D Generation".

<p align="center">
  <img src="assets/demo.gif" alt="Supplementary video demo" width="100%">
</p>

## Authors

Jiazhong Cen<sup>1</sup><sup>,2</sup>,
Jiemin Fang<sup>2</sup><sup>,✉</sup>,
Sikuang Li<sup>1</sup><sup>,2</sup>,
Guanjun Wu<sup>3</sup><sup>,2</sup>,
Chen Yang<sup>2</sup>,
Taoran Yi<sup>3</sup><sup>,2</sup>,
Zanwei Zhou<sup>1</sup><sup>,2</sup>,
Zhikuan Bao<sup>2</sup>,
Lingxi Xie<sup>2</sup>,
Wei Shen<sup>1</sup><sup>,✉</sup>,
Qi Tian<sup>2</sup>

<sup>1</sup> MoE Key Lab of Artificial Intelligence, AI Institute, School of Computer Science, Shanghai Jiao Tong University  
<sup>2</sup> Huawei Inc.  
<sup>3</sup> Huazhong University of Science and Technology

Contact: jaminfong@gmail.com, wei.shen@sjtu.edu.cn

## Overview

TIGON is a text-image conditioned 3D generation framework that supports:

- text-to-3D generation
- image-to-3D generation
- interleaved text-image conditioned 3D generation

The repository currently provides the inference pipeline and demo entry for interactive generation.

## Installation

### 1. Create the environment

Please create the runtime environment from [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate tigon
```

### 2. Install extra dependencies

After the base environment is ready, create an `external` directory under the repository root and install the required external dependencies:

```bash
mkdir -p external
cd external

git clone https://github.com/autonomousvision/mip-splatting.git
pip install mip-splatting/submodules/diff-gaussian-rasterization --no-build-isolation

pip install flash-attn --no-build-isolation

git clone https://github.com/NVlabs/nvdiffrast.git
pip install ./nvdiffrast --no-build-isolation

git clone https://github.com/facebookresearch/dinov3.git
```

Then place the DINOv3 ViT-H/16+ checkpoint at:

```bash
./external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

### 3. Compatibility note

The environment used by TIGON is the same as the environments used by [TRELLIS](https://github.com/Microsoft/TRELLIS) and [UniLat3D](https://github.com/UniLat3D/UniLat3D). If you have already prepared either of those environments, you can directly use this repository in most cases.

You still need to make sure the extra components required by TIGON are correctly prepared, especially:

- CLIP-related dependencies in the environment
- DINOv3 codebase and weight file under `external`


## Checkpoints

The pretrained checkpoint is available at the Hugging Face repository below:

- https://huggingface.co/JumpCat/TIGON

After downloading the checkpoint, place the `mix_e2e_pipe` folder under the repository root:

```bash
tigon/
|-- mix_e2e_pipe/
|-- demo.py
|-- trellis/
|-- configs/
|-- ...
```

The demo script loads the checkpoint from:

```bash
./mix_e2e_pipe
```

## Inference

After the environment and checkpoint are ready, run:

```bash
python demo.py
```

The script supports three generation modes:

- text only
- image only
- text + image interleaved conditioning

During execution, the script will ask for:

- random seed
- text prompt
- image path

Generated results are saved under `interactive_output/`, including:

- rendered 3D video in `.mp4`
- four-view rendered images in `.png`
- input metadata in `_info.txt`
- saved reference condition image in `_ref.png`

## Notes

- `demo.py` defaults to `CUDA_VISIBLE_DEVICES=0`.
- The script enables pipeline offloading by default through `TIGON_ENABLE_OFFLOAD=1`.
- The checkpoint is expected to provide the `gaussian` output format for rendering and visualization.

## Repository Structure

```text
TIGON/
|-- demo.py
|-- environment.yml
|-- configs/
|-- trellis/
|-- condition_images/
|-- external/               # created manually during setup
|-- mix_e2e_pipe/           # downloaded checkpoint folder
```

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{cen2026tigon,
  title     = {Text-Image Conditioned 3D Generation},
  author    = {Cen, Jiazhong and Fang, Jiemin and Li, Sikuang and Wu, Guanjun and Yang, Chen and Yi, Taoran and Zhou, Zanwei and Bao, Zhikuan and Xie, Lingxi and Shen, Wei and Tian, Qi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## Acknowledgement

This project builds upon the codebase and environment foundations of TRELLIS and UniLat3D. We thank the authors of these projects for making their work available.
