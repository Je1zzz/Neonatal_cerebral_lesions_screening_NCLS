# Neonatal Cerebral Lesions Screening (NCLS)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub](https://img.shields.io/badge/GitHub-Link-blue)](https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS)


🚀 Demo of Neonatal Cerebral Lesions Screening (NCLS).

## 1. Clone the Repository

First, clone the project to your local machine. Open your terminal and run the following commands:

```
git clone https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS.git
cd Neonatal_cerebral_lesions_screening_NCLS
```

## 2. Set Up the Environment and Download Data

### 2.1 Create Environment and Install Dependencies

```
conda create --name NCLS python=3.10
conda activate NCLS
pip install -r requirements.txt
```

### 2.2 Download Data and Weights

Download the model weights and save them to the `./log` folder

Download the example data (video data) and save them to the `./Example_` folder

Download link : [Demo data and model weight download LINK](https://drive.google.com/drive/folders/1aQDuLPmSBAULJ5soqeizaEkAHiwfpV1o?usp=sharing)
## 3. Run the `module_diagnosis.py` File

```
python module_diagnosis.py
```

Several steps will be taken next: 

1. Automatically extract standard views from CUS (cranial ultrasound) videos and save to the 'output/Standard View' folder.
2. Diagnose whether each newborn has severe brain injury based on the standard views. All results will save to 'output/Diagnostic result' folder. 