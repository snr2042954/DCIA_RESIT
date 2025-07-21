# ALS-Based Student Modeling and Microgoal Recommendation System

This project applies Alternating Least Squares (ALS) matrix factorization to model student performance on educational microgoals. It supports hyperparameter optimization, evaluation against a baseline, and personalized recommendations that identify which tasks a student may safely skip based on predicted mastery.

## Project Overview

The pipeline includes:

1. Data preprocessing and filtering
2. ALS model training with cross-validated hyperparameter tuning
3. Model evaluation using RMSE and classification metrics
4. Personalized recommendations for students based on predicted ability

## File Descriptions

### `data_selection.py`

Prepares the student-microgoal matrix for modeling.

- Replaces `-2` with missing values (NaNs)
- Filters out rows with illogical values
- Clips scores to [-2, 2]
- Selects the top `n_students` with the most non-missing data

**Output:**  
- `../Data/shortened_student_abilities.csv`

---

### `ALS_grid_search.py`

Trains ALS models with different combinations of latent factors and iteration counts using cross-validation. Evaluates RMSE to identify the optimal configuration.

- Implements the ALS training function (`train_als`)
- Performs grid search over latent dimensions and iterations
- Plots a heatmap of RMSE scores

**Output:**  
- `../Data/best_params.json`

---

### `ALS_evaluation.py`

Evaluates ALS model performance by comparing predictions to ground truth values and to a baseline that uses global mean imputation.

- Loads best ALS parameters from JSON
- Computes RMSE, accuracy, precision, recall, and F1-score
- Plots prediction error distribution and confusion matrix
- Samples and displays a few held-out predictions

**Output:**  
- Does not output anything perse but does print evaluation and diagnostic plots in console

---

### `ALS_recommendations.py`

Generates personalized microgoal recommendations for a selected student.

- Loads student-microgoal matrix and microgoal metadata
- Trains ALS using best parameters
- Recommends microgoals from known topic "islands" where the student is predicted to perform well (ELO ≥ 1.0)
- These tasks are considered already mastered and may be skipped

**Output:**  
- Does not output anything per se but does print recommendation list for a student in console


## Prediction Interpretation

ALS predicts personalized ELO-style scores for each student–microgoal pair:

- ELO ≥ 1.0: Student is highly likely to succeed
- ELO ≈ 0: Uncertain or average ability
- ELO < 0: Student may struggle

In this project, ELO ≥ 1.0 is used as the threshold for **skip-worthy recommendations**, assuming the student has already mastered the concept.

## Running the Project

1. **Preprocess the data:**

Run the script in `data_selection.py` to preprocess the data.

2. **Run ALS grid search and save best parameters:**

Run the script in `ALS_grid_search.py` to output the best parameters for ALS construction.

   (Optionally also run `ALS_evaluation.py` to evaluate the performance of the model)

3. **Generate student-specific recommendations:**

Run `ALS_recommendations.py` to get the recommendations for microgoals to skip