当然可以！下面是**完整的原始 Markdown 源码版本**的 `README.md`，你可以直接复制粘贴到 GitHub，**所有加粗的部分都使用 `**` 标出，完全保留 Markdown 语法标记**，不会被渲染成粗体。

---

```markdown
# 🛰️ LBP-SVM: Aerial Landscape Image Classification

This repository contains an implementation of aerial image classification using **Local Binary Pattern (LBP)** features and an **SVM classifier**.

We explore three variants:

- ✅ **LBP-SVM_original.ipynb**: Baseline with full dataset  
- 🧪 **LBP-SVM_augmented.ipynb**: With data augmentation  
- ⚖️ **LBP-SVM_imbalanced.ipynb**: With artificially imbalanced dataset

---

## 📂 Dataset

- **Name:** `Aerial_Landscapes`
- **Classes (15):**  
  `Agriculture`, `Airport`, `Beach`, `City`, `Desert`, `Forest`, `Grassland`, `Highway`, `Lake`, `Mountain`, `Parking`, `Port`, `Railway`, `Residential`, `River`
- Each class originally has **800 grayscale images**
- Resized to **128x128 pixels**

---

## 🧠 Method Overview

| **Component**         | **Details**                                      |
|----------------------|--------------------------------------------------|
| Feature Extractor     | Local Binary Pattern (LBP)                       |
| Classifier            | Support Vector Machine (SVM)                     |
| LBP Params            | `radius=1`, `n_points=8`, `method='uniform'`     |
| Evaluation            | Accuracy, Per-class performance, Confusion Matrix |

---

## 📊 Results & Visualizations

| **Version**     | **Confusion Matrix**                      | **Per-Class Accuracy Chart**                 |
|-----------------|-------------------------------------------|----------------------------------------------|
| Original        | ![conf_matrix](confusion_matrix.png)      | ![class_acc](per_class_accuracy.png)         |
| Augmented       | ![conf_matrix_aug](confusion_matrix_augmented.png) | ![class_acc_aug](per_class_accuracy_augmented.png) |
| Imbalanced      | ![conf_matrix_imb](confusion_matrix_imbalanced.png) | ![class_acc_imb](per_class_accuracy_imbalanced.png) |

---

## 🛠️ How to Run

### 🔧 Requirements

```bash
pip install numpy opencv-python scikit-learn matplotlib seaborn tqdm joblib
```

### 📁 Prepare the Dataset

> ⚠️ **Important:**  
> This repository **does not include the dataset** due to memory constraints.  
> Please manually place the `Aerial_Landscapes` dataset folder in the root directory:

```
LBP-SVM/
├── Aerial_Landscapes/
│   ├── Agriculture/
│   ├── Airport/
│   ├── ...
```

Each class should be in its own subfolder with `.jpg` or `.png` images.

### 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/LBP-SVM.git
cd LBP-SVM
```

2. Place your dataset as shown above.

3. Run any of the notebooks in Jupyter:

```text
LBP-SVM_original.ipynb      # Baseline model
LBP-SVM_augmented.ipynb     # With data augmentation
LBP-SVM_imbalanced.ipynb    # With class imbalance
```

Trained models (`.joblib`) and plots will be automatically saved.

---

## 📁 File Structure

```
📁 LBP-SVM/
├── LBP-SVM_original.ipynb
├── LBP-SVM_augmented.ipynb
├── LBP-SVM_imbalanced.ipynb
├── lbp_svm_model*.joblib
├── label_encoder*.joblib
├── confusion_matrix*.png
├── per_class_accuracy*.png
```

---

## 📬 Contact

For issues, questions, or suggestions, feel free to open an issue or reach out!

> Created with ❤️ by **[Your Name]**
```

---

✅ 复制这段内容直接粘贴到 GitHub 上的 `README.md` 中，就会正确渲染出你想要的加粗标题、表格、代码块等。

需要我帮你自动加徽章、加中文版本或者一键部署说明也可以继续说～
