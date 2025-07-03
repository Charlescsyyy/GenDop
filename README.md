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
[2025-07-01] Released training code.

<!-- [2025-07-08] Released the curated trajectory dataset DataDoP along with its construction code. -->
<!-- [2025-07-15] Launched the Gradio demo. -->

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


## 📚 Dataset
We provide [DataDoP](https://huggingface.co/datasets/Dubhe-zmc/DataDoP), a large-scale multi-modal dataset containing 29K realworld shots with free-moving camera trajectories, depth maps, and detailed captions in specific movements, interaction with the scene, and directorial intent. 

Currently, we are releasing a subset of the dataset for validation purposes. Additional data will be made available coming soon.

<!-- Please refer to the dataset README for more details. -->


## 🏋️‍♂️ Training

**Note:**  We have released a subset of the [DataDoP](https://huggingface.co/datasets/Dubhe-zmc/DataDoP) dataset for training and validation. Please organize your training data in the following structure. If you wish to use your own dataset, refer to our data format or modify the [core/provider.py](./core/provider.py) file as needed.

```
GenDoP
├── DataDoP
│   ├── train
│   ├── test
│   ├── train_valid.txt
│   ├── test_valid.txt
```

**Training Commands**  
- Text (motion)-to-trajectory:
  ```bash
  accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_motion' --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77
  ```
- Text (directorial)-to-trajectory:
  ```bash
  accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_directorial' --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77
  ```
- Text & RGBD-to-trajectory:
  ```bash
  accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_rgbd' --cond_mode 'depth+image+text' --text_key 'Concise Interaction' --num_cond_tokens 591
  ```

**Training Details**  
The model is trained on a single A100 (80GB) GPU for approximately 8 hours, with a batch size of 16, using a dataset of 30k examples for around 100 epochs.
- Recommended hyperparameters:
  ```
  --discrete_bins 256 --pose_length 30 --hidden_dim 1024 --num_heads 8 --num_layers 12
  ```
You can adjust these parameters in [core/options.py](./core/options.py) according to your specific requirements.

## 📆 Todo
<!-- - [ ] Release Inference Code  -->
- [ ] Release Dataset
- [ ] Release Dataset Construction Code
- [ ] Gradio Demo

## 📚 Acknowledgements
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
- [EdgeRunner](https://github.com/NVlabs/EdgeRunner)
- [E.T.](https://github.com/robincourant/the-exceptional-trajectories)

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
