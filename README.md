<h1 align="center">NCLS：新生儿脑损伤筛查系统</h1>
<h3 align="center">Neonatal Cerebral Lesions Screening</h3>

<p align="center">
  <a href="https://www.nature.com/articles/s41467-025-63096-9"><img src="https://img.shields.io/badge/Nature%20Communications-Paper-0b7fab.svg" alt="Nature Communications"></a>
  <a href="https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS/stargazers"><img src="https://img.shields.io/github/stars/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS" alt="Stars"></a>
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c.svg" alt="PyTorch 2.2.0">
</p>

---

## 📰 最新消息

**[2025-08-21]** 论文 **"Deep learning approach for screening neonatal cerebral lesions on ultrasound in China"** 已发表于 [*Nature Communications*](https://www.nature.com/articles/s41467-025-63096-9)

---

## 📖 项目简介

NCLS 是一个基于深度学习的新生儿颅脑超声自动筛查系统，可以：
- ✨ **自动提取标准视图**：从颅脑超声视频中智能识别并提取标准视图
- 🔍 **智能诊断**：基于提取的标准视图自动判断是否存在严重脑损伤
- ⚡ **高效准确**：结合目标检测和分类模型，实现快速准确的筛查

<details>
  <summary><b>📊 查看系统架构</b></summary>
  <p align="center">
    <img src="./output/Figure1_01.png" alt="Overall Architecture" width="100%">
  </p>
</details>

---

## 🚀 快速开始

### 环境配置

本项目基于 **Python 3.12**、**PyTorch 2.2.0** 和 **torchvision 1.17.0** 开发。

```bash
# 克隆仓库
git clone https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS.git
cd Neonatal_cerebral_lesions_screening_NCLS

# 创建 conda 环境
conda create --name NCLS python=3.12
conda activate NCLS

# 安装依赖
pip install -r requirements.txt
```

> **💡 提示**：关于 CUDA 和 PyTorch 版本对应关系，请参考：
> - [CUDA与GPU对应关系](https://zhuanlan.zhihu.com/p/633473214)
> - [PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)

### 下载数据和模型

从 [Google Drive](https://drive.google.com/drive/folders/1aQDuLPmSBAULJ5soqeizaEkAHiwfpV1o?usp=sharing) 下载：
- 📦 **示例数据**：放置到 `./Example_` 文件夹
- 🎯 **预训练权重**：放置到 `./log` 文件夹

```
项目结构：
Neonatal_cerebral_lesions_screening_NCLS/
├── Example_/          # 示例视频数据
├── log/               # 模型权重文件
│   ├── diagnostic_weight/
│   └── detection_weight.pth
├── output/            # 输出结果
├── configs/           # 配置文件
├── models/            # 模型定义
└── utils/             # 工具函数
```

---

## 🎯 运行推理

### 完整推理流程

在命令行中运行以下命令，系统将自动完成视图提取和诊断：

```bash
python module_diagnosis.py \
    --cfg_classfication configs/convnext.yaml \
    --weight_classfication log/diagnostic_weight \
    --cfg_detection configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
    --weight_detection log/detection_weight.pth \
    --dicom-dir Example_ \
    --output-dir output \
    --device cuda
```

### 仅提取标准视图

如果只需要提取标准视图：

```bash
python module_extract_view.py \
    --cfg_detection configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
    --weight_detection log/detection_weight.pth \
    --dicom-dir Example_ \
    --output-dir output \
    --device cuda
```

### 执行流程

1. **视图提取**：从颅脑超声视频中自动提取标准视图，保存到 `output/StandardViews/` 文件夹
2. **诊断分析**：基于提取的标准视图进行诊断，结果保存到 `output/DiagnosisResult/` 文件夹

---

## 📊 可视化结果

### 提取的标准视图示例

<p align="center">
  <img src="output/extracted.png" width="85%" alt="Standard Views Example">
</p>

### 诊断结果示例

<p align="center">
  <img src="./output/result.png" width="85%" alt="Diagnostic Result">
</p>

---

## 🙏 致谢

本项目使用了 [RT-DETR](https://github.com/lyuwenyu/RT-DETR) 框架进行实时目标检测。感谢作者开源代码。为了保持项目简洁，我们仅保留了推理所需的代码。如需完整代码，请参考原始仓库。

---

## 📝 引用

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@article{lin2025deep,
  title   = {Deep learning approach for screening neonatal cerebral lesions on ultrasound in China},
  author  = {Lin, Zhouqin and Zhang, Haoming and Duan, Xingxing and Bai, Yan and Wang, Jian and 
             Liang, Qianhong and Zhou, Jingran and Xie, Fusui and Shentu, Zhen and Huang, Ruobing and 
             Chen, Yayan and Yu, Hongkui and Weng, Zongjie and Ni, Dong and Liu, Lei and Zhou, Luyao},
  journal = {Nature Communications},
  volume  = {16},
  number  = {1},
  pages   = {7778},
  year    = {2025},
  doi     = {10.1038/s41467-025-63096-9}
}
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

<p align="center">
  Made with ❤️ by the NCLS Team
</p>
