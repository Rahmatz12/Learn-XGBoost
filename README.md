# Learning XGBoost

A small, reproducible starting point for learning gradient-boosted trees with XGBoost. The starter trains a binary classifier on scikit-learn's built-in breast cancer dataset, so it does not need a data download.

## Set up

In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the example:

```powershell
python train_model.py
```

The script reports accuracy and ROC AUC on a stratified test split. Try changing `max_depth`, `learning_rate`, and `n_estimators`, then compare the results.