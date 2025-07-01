# GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography [ICCV 2025]


<p align="center">
<a href="https://arxiv.org/abs/2504.07083"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://kszpxxzmc.github.io/GenDoP/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://www.youtube.com/watch?v=UWvR_A7yFeI"><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<a href=""><img src="https://img.shields.io/static/v1?label=Dataset&message=Data&color=yellow"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=TODO" />
</a>
</p>

[**Paper**](https://arxiv.org/abs/2504.07083) | [**Project page**](https://kszpxxzmc.github.io/GenDoP/) | [**Video**](https://www.youtube.com/watch?v=UWvR_A7yFeI) | [**Data**]() 

[Mengchen Zhang](https://kszpxxzmc.github.io), [Tong Wu✉️](https://wutong16.github.io), [Jing Tan](https://sparkstj.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Gordon Wetzstein](https://stanford.edu/~gordonwz/), [Dahua Lin✉️](http://dahua.site/)

![Teaser Image](./assets/teaser.png)


## ✨ Updates
[2025-07-01] Released inference code and model checkpoints!

<!-- [2025-07-08] Released the curated trajectory dataset DataDoP along with its construction code.

[2025-07-15] Released training code.

[2025-07-15] Launched the Gradio demo. -->

## 📦 Install 
Make sure ```torch``` with CUDA is correctly installed. For training, we rely on ```flash-attn``` (requires at least Ampere GPUs like A100). For inference, older GPUs like V100 are also supported, although slower.
```
git clone https://github.com/3DTopia/GenDoP.git
cd GenDoP

conda create --name GenDoP python=3.10
conda activate GenDoP
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```
## 💡 Inference 

### Pretrained Models

### Trajectory Generation

### Visualization


<!-- ## 📚 Dataset

## 🏋️‍♂️ Training -->

## 📆 Todo
- [ ] Release Inference Code 
- [ ] Release Dataset
- [ ] Release Dataset Construction Code
- [ ] Release Training Code
- [ ] Gradio Demo

## 📚 Acknowledgements
Special thanks to [EdgeRunner](https://github.com/NVlabs/EdgeRunner), [E.T.](https://github.com/robincourant/the-exceptional-trajectories) for codebase.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@misc{zhang2025gendopautoregressivecameratrajectory,
      title={GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography}, 
      author={Mengchen Zhang and Tong Wu and Jing Tan and Ziwei Liu and Gordon Wetzstein and Dahua Lin},
      year={2025},
      eprint={2504.07083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07083}, 
}
```
