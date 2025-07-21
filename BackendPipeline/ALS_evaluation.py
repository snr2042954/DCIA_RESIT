## EVALUATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import json
from ALS_grid_search import train_als

def baseline_comparison(values, mask_full, coords, mu, best_rmse, hide_ratio=0.2, seed=42):
    """
    Compares the ALS model to a mean baseline using held-out data.

    args:
        values (ndarray): Full rating matrix.
        mask_full (ndarray): Full boolean mask.
        coords (list of tuples): (user, item) indices of known values.
        mu (float): Global mean.
        best_rmse (float): Best ALS RMSE for comparison.
        hide_ratio (float): Proportion of known values to hide for testing.
        seed (int): Random seed for reproducibility.

    returns:
        mask_demo, mask_test_demo (bool arrays): Masks used for training and testing.
    """
    mask_demo = mask_full.copy()
    mask_test_demo = np.zeros_like(mask_full, dtype=bool)

    coords_demo = coords.copy()
    np.random.seed(seed)
    np.random.shuffle(coords_demo)

    n_hide_demo = int(len(coords_demo) * hide_ratio)
    for (u, v) in coords_demo[:n_hide_demo]:
        mask_demo[u, v] = False
        mask_test_demo[u, v] = True

    baseline_preds = np.full_like(values, mu)
    diffs_baseline = baseline_preds[mask_test_demo] - values[mask_test_demo]
    baseline_rmse = np.sqrt(np.mean(diffs_baseline ** 2))
    improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100

    print(f"Optimal parameters: n_factors={best_f}, n_iters={best_it}, CV RMSE={best_rmse:.4f}")
    print(f"Baseline RMSE (mean imputation): {baseline_rmse:.4f}")
    print(f"Improvement over baseline: {improvement:.1f}% lower RMSE\n")

    return mask_demo, mask_test_demo

def classification_evaluation(values, preds, mask_test, threshold=1):
    """
    Evaluates binary classification performance at a given threshold.

    args:
        values (ndarray): Ground truth values.
        preds (ndarray): Predicted values.
        mask_test (bool ndarray): Boolean mask for test entries.
        threshold (float): Threshold for binarizing predictions.

    output:
        Prints classification metrics and shows confusion matrix.
    """
    true_bin = (values[mask_test] > threshold).astype(int)
    pred_bin = (preds[mask_test] > threshold).astype(int)

    accuracy = accuracy_score(true_bin, pred_bin)
    precision = precision_score(true_bin, pred_bin)
    recall = recall_score(true_bin, pred_bin)
    f1 = f1_score(true_bin, pred_bin)

    print(f"\nClassification metrics (threshold={threshold}):")
    print(f"  Accuracy:  {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1 Score:  {f1:.2f}")

    cm = confusion_matrix(true_bin, pred_bin)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Threshold = {threshold})')
    plt.show()

def plot_error_distribution(preds, values, mask_test):
    """
    Plots the distribution of raw prediction errors.

    args:
        preds (ndarray): Predicted values from ALS.
        values (ndarray): Ground truth values.
        mask_test (bool ndarray): Mask for test entries.
    """
    errors = preds[mask_test] - values[mask_test]

    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=400, alpha=0.7, color='purple')
    plt.title('Distribution of Prediction Error (ELO scores)')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.xlim(-2.5, 2.5)
    plt.xticks(np.linspace(-2.5, 2.5, num=21))
    plt.grid(True)
    plt.show()

def get_ground_truth_comparison(df, values, preds, coords, n_samples=5):
    """
    Creates a DataFrame comparing true and predicted ELO scores for a set of observed entries.

    args:
        df (DataFrame): Original DataFrame with student IDs as index and microgoals as columns.
        values (ndarray): Ground truth values (2D NumPy array from df).
        preds (ndarray): Predicted values (same shape as values).
        coords (list of tuples): List of (row, col) indices for observed (non-missing) values.
        n_samples (int): Number of samples to include in the output (default = 5).

    returns:
        predictions_df (DataFrame): A DataFrame with user, microgoal, true value, prediction, and error.
    """
    results = []

    for i, (u, v) in enumerate(coords[:n_samples], start=1):
        user = df.index[u]
        microgoal = df.columns[v]
        true_score = values[u, v]
        pred_score = preds[u, v]

        results.append({
            "Sample #": i,
            "User": user,
            "Microgoal": microgoal,
            "True ELO": round(true_score, 2),
            "Predicted ELO": round(pred_score, 2),
            "Error": round(abs(pred_score - true_score), 2)
        })

    predictions_df = pd.DataFrame(results)
    return predictions_df

if __name__ == "__main__":

    # Load the data
    df_students = pd.read_csv('../Data/shortened_student_abilities.csv', index_col=0)

    # Convert DataFrame to NumPy array and identify non-missing values
    values = df_students.values.astype(float)
    mask_full = ~np.isnan(values)
    mu = values[mask_full].mean()
    coords = list(zip(*np.where(mask_full)))

    # Load the best parameters from the JSON file
    with open('../Data/best_params.json', 'r') as f:
        best_params = json.load(f)
    best_f = best_params["best_factors"]     # int: best number of latent factors
    best_it = best_params["best_iters"]      # int: best number of ALS iterations
    best_rmse = best_params["best_rmse"]     # float: best RMSE value
    min_idx = tuple(best_params["best_index"])  # tuple: index in RMSE grid

    # Run baseline comparison and get evaluation masks
    mask_demo, mask_test_demo = baseline_comparison(
        values, mask_full, coords, mu, best_rmse, hide_ratio=0.2, seed=42
    )

    # Train ALS model with the best parameters on demo set
    R_train_demo = np.where(mask_demo, values, mu) - mu
    U_demo, V_demo = train_als(R_train_demo, mask_demo, n_factors=best_f, n_iters=best_it, reg=0.1)
    preds_demo = U_demo @ V_demo.T + mu

    # Evaluate classification performance
    classification_evaluation(values, preds_demo, mask_test_demo, threshold=1)

    # Visualize prediction error distribution
    plot_error_distribution(preds_demo, values, mask_test_demo)

    # Print a few examples from the held-out set
    comparison_df = get_ground_truth_comparison(df_students, values, preds_demo, coords, n_samples=5)
    print("Sample held-out predictions:")
    print(comparison_df)