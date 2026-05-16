# NGBoost-BridgeSeismicAssessment

A machine learning framework for **seismic damage assessment of bridges** using NGBoost.  
This project develops **probabilistic models** to predict bridge damage states under earthquake ground motions, supporting fragility analysis and post-earthquake risk evaluation.

---

## 📄 Paper Reference

**Title:** "Rapid assessment of bridge post-earthquake response using unsupervised clustering and probabilistic machine learning under large-scale Kik-Net data"  
**DOI:** *[Add DOI here]*  

---

## 📁 Directory Structure

- NGBoost-BridgeSeismicAssessment/
  - 📂 code/
    - 📁 best_model/       Saved trained NGBoost models
    - 📁 ngboost_pred/     Scripts for prediction using NGBoost
    - 📁 ngboost_train/    Scripts for training NGBoost models
  - 📂 data/               Training and testing datasets

## 🖥️ Computational Environment

### System Requirements
| Item | Specification |
|------|--------------|
| Operating System | Windows 10 (10.0.19041) or later |
| Processor | Intel64 Family 6 or equivalent |
| Python | 3.8.8 |

### Required Dependencies
| Package | Version |
|---------|---------|
| numpy | 1.23.5 |
| pandas | 1.4.4 |
| scipy | 1.8.0 |
| scikit-learn | 1.3.2 |
| ngboost | 0.4.2 |
| xgboost | 2.0.3 |
| statsmodels | 0.14.1 |
| matplotlib | 3.1.2 |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |
| joblib | 1.3.2 |

### Installation
Create a conda environment and install dependencies:

```bash
# Create conda environment
conda create -n ngboost_bridge python=3.8.8
conda activate ngboost_bridge

# Install required packages
pip install numpy==1.23.5
pip install pandas==1.4.4
pip install scipy==1.8.0
pip install scikit-learn==1.3.2
pip install ngboost==0.4.2
pip install xgboost==2.0.3
pip install statsmodels==0.14.1
pip install matplotlib==3.1.2
pip install seaborn==0.13.2
pip install openpyxl==3.1.5
pip install joblib==1.3.2
```
