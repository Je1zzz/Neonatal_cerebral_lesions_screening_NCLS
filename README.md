<h1 align="center">NCLS : Neonatal Cerebral Lesions Screening</h1>


<p align="center">
  <a href="https://www.nature.com/articles/s41467-025-63096-9"><img src="https://img.shields.io/badge/Nature%20Communications-文章-0b7fab.svg" alt="Nature Communications"></a>
  <a href="https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS/stargazers"><img src="https://img.shields.io/github/stars/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS" alt="Stars"></a>
</p>

<p align="left">📣 <b>News</b>: [2025-08-21] Paper <b><u>Deep learning approach for screening neonatal cerebral lesions on ultrasound in China</u></b> published in <a href="https://www.nature.com/articles/s41467-025-63096-9">Nature Communications</a></p>

<details>
  <summary>Fig: Overall Architecture</summary>
  <p align="center">
    <img src="./output/Figure1_01.png" alt="Overall Architecture" width="100%">
  </p>
</details>

## 🍔 Usage


1. This project is implemented with **Python 3.12**, **torch 2.2.0**, and **torchvision 1.17.0**. You can find the relationship between CUDA and GPU at [Zhihu](https://zhuanlan.zhihu.com/p/633473214), and find the corresponding installation packages at [PyTorch_Previous_Versions](https://pytorch.org/get-started/previous-versions/).

   ```bash
   ## clone repo
   git clone https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS.git
   cd Neonatal_cerebral_lesions_screening_NCLS

   ## create conda env
   conda create --name NCLS python=3.10
   conda activate NCLS

   ## install dependencies
   pip install -r requirements.txt
   ```

2. Here you can download a few example data and pretrained weights of two models: [data&weight](https://drive.google.com/drive/folders/1aQDuLPmSBAULJ5soqeizaEkAHiwfpV1o?usp=sharing). Make sure to put the weights to the `./log` folder, while the videos are put to the `./Example_` folder.


## 🍨 Testing

1. Run on the command line:
    ```bash
    python module_diagnosis.py \
        --cfg_classfication configs/convnext.yaml \
        --weight_classfication log/diagnostic_weight.pth \
        --cfg_detection configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
        --weight_detection log/detection_weight.pth \
        --dicom-dir Example_ \
        --output-dir output \
        --device cuda
    ```

2. The following steps will be performed:

    - Automatically extract standard views from CUS (cranial ultrasound) videos and save them to the 'output/Standard View' folder.
    - Diagnose whether each newborn has severe brain injury based on the standard views. All results will be saved to the 'output/Diagnostic result' folder.

## 🍼 Visualization

#### 🍭 Extracted Standard Views of a Single Case

<p align="left">
  <img src="output/extracted.png" width="80%" alt="Standard Views Example">
</p>

#### 🍰 Diagnostic Result
![Diagnostic Result](./output/result.png)

## ☕ Acknowledgements

This project uses the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) framework for real-time detection. We thank the authors for making their code open source.

## Citation
If you find our work useful, please cite:
```bibtex
@article{lin2025deep,
  title   = {Deep learning approach for screening neonatal cerebral lesions on ultrasound in China},
  author  = {Lin, Zhouqin and Zhang, Haoming and Duan, Xingxing and Bai, Yan and Wang, Jian and Liang, Qianhong and Zhou, Jingran and Xie, Fusui and Shentu, Zhen and Huang, Ruobing and Chen, Yayan and Yu, Hongkui and Weng, Zongjie and Ni, Dong and Liu, Lei and Zhou, Luyao},
  journal = {Nature Communications},
  volume  = {16},
  number  = {1},
  pages   = {7778},
  year    = {2025},
  doi     = {10.1038/s41467-025-63096-9},
}
```