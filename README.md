# K-Nearest Neighbors (KNN) Classifier - Wine Dataset

This project demonstrates a from-scratch implementation of the **K-Nearest Neighbors (KNN)** algorithm using the Wine dataset from the UCI Machine Learning Repository.

---

## 📁 Project Structure

```
.
├── knn.py             # Core KNN algorithm and distance functions
├── analysis.ipynb     # Data processing, model evaluation, and visualization
```

---

## 📊 Dataset

**Wine Dataset**  
- Contains 178 samples  
- 13 numerical features  
- 3 wine classes

The dataset should be saved as `wine.data` and is loaded inside the notebook.

---

## ⚙️ Features

- Custom KNN implementation (no ML libraries)
- Supported distance metrics:
  - Euclidean
  - Manhattan
  - Minkowski (p=3)
- Accuracy evaluation
- Accuracy vs K-value visualization
- Confusion matrix with per-class precision, recall, and F1-score

---

## 🚀 How to Use

### 1. Using Google Colab or Jupyter Notebook
- Firstl, install the dependencies
- The notebook imports `knn.py` to use the algorithm functions
- You have to load `knn.py` file to where you will run the code
- Open and run `analysis.ipynb`


### 2. In a local Python environment
Run the notebook via Jupyter:

```bash
jupyter notebook analysis.ipynb
```

---

## 📦 Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

Install them with:

```bash
pip install numpy pandas matplotlib seaborn
```

---
