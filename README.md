# Breast Cancer Diagnosis with a Deep Learning MLP

Binary classification of breast tumours (benign vs. malignant) from 30 cell-nucleus measurements, using a Keras
multi-layer perceptron, with seven-stage data visualization (Fry's Acquire, Parse, Filter, Mine, Represent, Refine, Interact).

**Author:** Sredivit K T (24AD112), B.Tech AI & DS, 3rd Year, KPR Institute of Engineering and Technology

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17
- 569 samples, 30 numeric features (10 nucleus properties x mean / standard error / worst), 357 benign and 212 malignant.
- The notebook loads it with `sklearn.datasets.load_breast_cancer`, so no separate download is needed.

## What is in this repository
| Path | Contents |
|---|---|
| `breast_cancer_dl_project.ipynb` | Full code with outputs: data loading, EDA, preprocessing, 7 visualizations, model, tuning, evaluation |
| `report/Project_Report.pdf` and `.docx` | Project report |
| `figures/` | All generated charts, including `v7_interactive_scatter.html` (open in a browser) |
| `requirements.txt` | Python dependencies |

## Method in brief
1. **EDA and preprocessing:** no missing values or duplicates; extreme outliers (beyond 3 x IQR) capped using training-set bounds; features standardized; stratified 70/15/15 split.
2. **Visualization:** class distribution, histograms, boxplots, correlation heatmap, pairplot, refined scatter, interactive scatter.
3. **Model:** MLP 30 - 32 - 16 - 1 (batch normalization, dropout 0.1, L2), Adam (lr 0.005), early stopping; 12-point grid search on validation loss. Logistic regression baseline.
4. **Evaluation:** loss/accuracy curves, confusion matrix, ROC and AUC, error distribution.

## Results (held-out test set, 86 samples)
| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| MLP (Keras) | 0.9884 | 1.0000 | 0.9688 | 0.9994 |
| Logistic Regression | 0.9651 | 0.9677 | 0.9375 | 0.9988 |

The MLP made 1 false negative and 0 false positives. With only 86 test samples the 95% interval for accuracy is roughly 93.7% to 99.8%,
and the gap to the baseline is 2 samples, so these are single-split results and should be confirmed with cross-validation.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook breast_cancer_dl_project.ipynb
```
Run all cells (about 5 to 15 minutes on a CPU; the tuning grid is the slowest part). Figures are saved to `figures/`.
Exact numbers can differ slightly across TensorFlow versions and hardware.

## Disclaimer
Educational project. It is not a diagnostic tool and must not be used for clinical decisions.
