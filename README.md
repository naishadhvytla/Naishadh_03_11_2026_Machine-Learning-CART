# Naishadh_03_11_2026_Machine-Learning-CART

CART (Classification and Regression Tree) implementation for a binary classification problem using a synthetic Loan Approval dataset.

## Files

- `CART_Loan_Approval.ipynb` — notebook with the full implementation and outputs
- `loan_approval_synthetic.csv` — the generated dataset (the notebook recreates it automatically)

## Dataset

40 applicants, 5 columns, generated in the notebook with a fixed random seed:

| Column | Type | Values |
|---|---|---|
| `Monthly_Income` | Numeric | 15–120 (thousand ₹) |
| `Credit_Score` | Numeric | 550–850 |
| `Employment_Type` | Categorical | Salaried / Self-Employed / Unemployed |
| `Owns_House` | Categorical | Yes / No |
| `Loan_Status` | Target | Approved (24) / Rejected (16) |

## Steps

1. Generate and explore the dataset
2. Compute Gini impurity, weighted child Gini and Gini reduction by hand
3. Search all candidate binary splits and find the best one
4. Train a CART model with `DecisionTreeClassifier(criterion="gini", max_depth=4)`
5. Evaluate with accuracy, classification report and confusion matrix
6. Visualize the tree and feature importance, and predict for new applicants

## Results

- Training accuracy: 96.88 %
- Test accuracy: 87.50 %
- Most important features: Credit_Score and Monthly_Income

## Run

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook CART_Loan_Approval.ipynb
```
