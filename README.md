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

  ## 📥 Download the Dataset

You can download the Wine dataset from the link below:

🔗 [Wine Data Set - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)

Import the file `wine.data` in your project directory.

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
- **Install the required dependencies.**
- **Download the Wine dataset** from [here](https://archive.ics.uci.edu/dataset/109/wine) 
- **Set the `file_path` variable** in the notebook to the relative path of the `wine.data` file.
- **Place the `wine.data` file** in the same folder as the notebook for convenience.
- The notebook **imports `knn.py`** to access the custom KNN algorithm functions.
- **Ensure `knn.py` is in the same directory** where you run the notebook.
- **Open and run `analysis.ipynb`.**



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
