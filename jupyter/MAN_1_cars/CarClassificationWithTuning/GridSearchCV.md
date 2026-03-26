# Understanding GridSearchCV for Hyperparameter Tuning

## What is GridSearchCV?

**GridSearchCV** is a powerful tool from **scikit-learn** used for **systematic hyperparameter exploration** and optimization.

It stands for:
- **Grid** → Tests every possible combination from a predefined grid of hyperparameters
- **Search** → Searches for the best performing combination
- **CV** → Cross-Validation (evaluates each combination reliably)

---

## Why Use GridSearchCV?

Machine learning models like Decision Trees and Random Forests have many **hyperparameters** that significantly impact performance:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `n_estimators` (Random Forest)
- `criterion`
- `class_weight`

These parameters greatly affect how well your model performs. Choosing them manually by trial and error is slow and unreliable.
GridSearchCV automates this process in a smart, systematic way.

Manually trying different combinations is time-consuming and unreliable.  
**GridSearchCV automates this process** in a structured and exhaustive way.

---

## How GridSearchCV Works (Step by Step)

1. **Define a Parameter Grid**

   You create a dictionary specifying which values to try for each hyperparameter.

   ```python
   dt_param_grid = {
       'max_depth': [3, 4, 5, 6, 8, 10],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4],
       'criterion': ['gini', 'entropy'],
       'class_weight': [None, 'balanced']
   }
   ```
   This example creates 72 possible combinations (2 × 3 × 3 × 2 × 2).



2. **Create the GridSearchCV Object**

    ```python
   dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='f1_weighted'    # Metric to optimize
)
    ```

3. **Fit the GridSearchCV**

    ```python
    dt_grid.fit(X_train, y_train)
    ```

- For every combination:
- The model is trained
- Performance is evaluated using cross-validation
- The average score is recorded

4. **Extract the Best Results**

```python
print("Best Parameters:", dt_grid.best_params_)
print("Best Cross-validated Score:", dt_grid.best_score_)

best_dt_model = dt_grid.best_estimator_
```

## What Happens Internally?

GridSearchCV performs an exhaustive search:

It generates all possible combinations of the parameters you defined.
For each combination, it runs k-fold cross-validation (here cv=5).
It calculates the chosen scoring metric (f1_weighted in this case).
After testing all combinations, it selects the one with the highest score.

### Simple Example

If you have:

max_depth: [3, 5, 8]
criterion: ['gini', 'entropy']

GridSearchCV will evaluate all 6 combinations and pick the best one based on your scoring metric.

### Why This Strategy Makes Sense in Your Case

Given your situation:

- Small dataset (~400 rows)
- Class imbalance (Origin 1 is majority class)
- Need for good generalization

Using GridSearchCV with:

- cv=5 (cross-validation)
- scoring='f1_weighted'
- class_weight options

helps you find hyperparameters that:

- Reduce overfitting
- Handle class imbalance effectively
- Generalize better on unseen data

### Key Advantages of GridSearchCV

- Systematic: Tries every combination (no guessing)
- Reliable: Uses cross-validation instead of a single train/test split
- Reproducible: Fixed random_state + clear grid
- Automated: Finds the best parameters for you

### Common Parameters Used in Your Code


| Parameter | Purpose | Value in Code |
|-------|-------|-------|
| estimator | The model to tune | DecisionTreeClassifier |
| param_grid | Dictionary of hyperparameters | dt_param_grid |
| cv | Number of cross-validation folds | 5 |
| scoring | Metric to optimize | 'f1_weighted' |

## Next Steps After GridSearchCV

Once you have the best model:

```python
best_dt_model = dt_grid.best_estimator_
y_pred = best_dt_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

You can then compare the fine-tuned model with the original (default) model.



