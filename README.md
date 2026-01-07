## ___***Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation***___
[Shaocong Xu](https://scholar.google.com/citations?user=PvYOrK0AAAAJ&hl=zh-CN), [Songlin Wei](https://songlin.github.io/), [Qizhe Wei](#), [Zheng Geng](#), [Hong Li](https://scholar.google.com/citations?user=5FBYzP8AAAAJ&hl=en), [Licheng Shen](#), [Qianpu Sun](https://scholar.google.com/citations?user=cVjaTlMAAAAJ&hl=zh-CN), [Shu Han](https://scholar.google.com/citations?user=qZGkubsAAAAJ&hl=zh-CN), [Bin Ma](#), [Bohan Li](https://scholar.google.com/citations?user=V-YdQiAAAAAJ&hl=zh-CN), [Chongjie Ye](https://scholar.google.com/citations?user=hP4G9iUAAAAJ&hl=en), [Yuhang Zheng](https://scholar.google.com/citations?user=Wn2Aic0AAAAJ&hl=en), [Nan Wang](https://scholar.google.com/citations?user=BWfLE6EAAAAJ&hl=zh-CN), [Saining Zhang](https://scholar.google.com/citations?user=P4efBMcAAAAJ&hl=zh-CN), and [Hao Zhao](https://scholar.google.com/citations?user=ygQznUQAAAAJ&hl=en)

<!-- <h3 align="center"> published conference</h3> -->

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2512.23705) 
[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://daniellli.github.io/projects/DKT/) 
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Daniellesry/DKT) 
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/collections/Daniellesry/dkt-models) 
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Dataset-blue)](https://huggingface.co/datasets/Daniellesry/TransPhy3D) 
[![YouTube](https://img.shields.io/badge/▶️%20YouTube-Video-red)](https://youtu.be/Vurjdwa_y38) 
[![ModelScope Model](https://img.shields.io/badge/🤖%20ModelScope-Model-orange)](https://modelscope.cn/collections/Daniellesry/DKT) 
[![ModelScope Dataset](https://img.shields.io/badge/🤖%20ModelScope-Dataset-orange)](https://modelscope.cn/datasets/Daniellesry/TransPhy3D/files) 
<a href="#"> <img src="https://visitor-badge.laobi.icu/badge?page_id=Daniellli.DKT" alt="Visitors"></a>
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0) 
</div>
 


## 🌟 Takeaways

DKT is a foundation model for **transparent-object 🫙**, **in-the-wild 🌎**, **arbitrary-length ⏳** video depth and normal estimation, facilitating downstream applications such as robot manipulation tasks, policy learning, and so forth.

![teaser](doc/main-teaser.png)

## ✨ News
- `[25-12-04]`  🔥🔥🔥 **DKT** is released now, have fun!




## 🤗 Pretrained Models
Our pretrained models are available on the huggingface hub:


<table>
  <thead>
    <tr>
      <th>Version</th>
      <th>Hugging Face Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DKT-Depth-1-3B</td>
      <td><a href="https://huggingface.co/Daniellesry/DKT-Depth-1-3B-v1.1" target="_blank"><code>DKT-Depth-1-3B-v1.1</code><a></td>
    </tr>
    <tr>
      <td>DKT-Depth-14B</td>
      <td><a href="https://huggingface.co/Daniellesry/DKT-Depth-14B" target="_blank"><code>DKT-Depth-14B</code><a></td>
    </tr>
    <tr>
      <td>DKT-Normal-14B</td>
      <td><a href="https://huggingface.co/Daniellesry/DKT-Normal-14B" target="_blank"><code>DKT-Normal-14B</code><a></td>
    </tr>
  </tbody>
</table>


## 📦 Installation
Please run following commands to build package:
```
git clone https://github.com/Daniellli/DKT.git
cd DKT
pip install -r requirements.txt
```


## 🤖 Gradio Demo

- Online demo: [DKT](https://huggingface.co/spaces/Daniellesry/DKT)
- Local demo: 
```
python app.py
```

## 💡 Usage

```
from dkt.pipelines.pipelines import DKTPipeline
import os
from tools.common_utils import save_video


pipe = DKTPipeline()

demo_path = 'examples/1.mp4'
prediction = pipe(demo_path,vis_pc = False)  #* Set vis_pc to `True` to obtain the estimated point cloud.


save_dir = 'logs'
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, 'demo.mp4')
save_video(prediction['colored_depth_map'], output_path, fps=25)



```



## 📜 Citation
```
@article{dkt2025,
  title   = {Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation},
  author  = {Shaocong Xu and Songlin Wei and Qizhe Wei and Zheng Geng and Hong Li and Licheng Shen and Qianpu Sun and Shu Han and Bin Ma and Bohan Li and Chongjie Ye and Yuhang Zheng and Nan Wang and Saining Zhang and Hao Zhao},
  journal = {https://arxiv.org/abs/2512.23705},
  year    = {2025}
}
```


## 💗 Ackownledge
Our code is based on recent fantastic works including [MoGe](https://github.com/microsoft/MoGe), [WAN](https://github.com/Wan-Video/Wan2.1), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). 
We sincerely thank the authors for their excellent contributions. Huge thanks!


## 📧 Contact
If you have any questions, please feel free to contact Shaocong Xu <b>(daniellesry at gmail.com)</b>.