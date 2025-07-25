# Decision Tree Classifier – Iris Species Prediction
This project implements a Decision Tree classifier to predict the species of iris flowers using the popular Iris dataset. It demonstrates core concepts in classification, decision tree modeling, model evaluation, and tree pruning.

## Demo

https://github.com/user-attachments/assets/3c5bc1fe-0266-4067-94b1-eca1e8b21080

## How to Run the Project
Clone the repository:
```bash
git clone https://github.com/Kiranwaqar/SpeciePrediction-DecisionTrees.git
cd Level_2_Task_2
```
Install dependencies:
```bash
pip install pandas scikit-learn matplotlib
```
Run the script:
```bash
python decisiontree.py
```
## Evaluation Metrics
- Accuracy: Measures overall correctness.
- F1 Score: Harmonic mean of precision and recall (great for imbalanced data).

## What is Tree Pruning?
Pruning is the process of reducing the size of the decision tree to avoid overfitting. It improves generalization by removing unnecessary branches.

You can control pruning using:

```python
DecisionTreeClassifier(max_depth=3)
```

