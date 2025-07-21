### IMPLEMENTING ALS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# Define all cross-validation settings in a dictionary
cross_val_settings = {
    "factors_list": [10, 20, 30],  # List of latent factor counts for ALS
    "iters_list": [10, 30, 50],    # List of iteration counts for ALS
    "n_folds": 2,                  # Number of cross-validation folds
    "reg": 0.01,                   # Regularization strength
    "random_seed": 42             # Seed for reproducibility
}

def train_als(R, mask, n_factors, n_iters, reg):
    """
    Trains a matrix factorization model using the Alternating Least Squares (ALS) algorithm.

    The goal is to approximate the original matrix R as a product of two lower-rank matrices U and V:
        R ≈ U @ V.T

    Missing values are handled by using a boolean mask to isolate observed entries.

    args:
        R (ndarray): 2D NumPy array (n_users x n_items), centered training matrix.
                     Missing values should be imputed before passing in (e.g., using mean).
        mask (ndarray): Boolean matrix (same shape as R), True where values are observed.
        n_factors (int): Number of latent factors to use for U and V.
        n_iters (int): Number of ALS iterations to perform.
        reg (float): Regularization strength (lambda) to prevent overfitting.

    returns:
        U (ndarray): User latent matrix of shape (n_users, n_factors).
        V (ndarray): Item latent matrix of shape (n_items, n_factors).
    """

    # Initialization
    n_users, n_items = R.shape  # Get matrix dimensions
    U = np.random.rand(n_users, n_factors)  # Initialize user factors randomly
    V = np.random.rand(n_items, n_factors)  # Initialize item factors randomly
    I_f = np.eye(n_factors)  # Identity matrix used in regularization

    # ALS Iterations
    for _ in range(n_iters):

        # Update user latent matrix U
        for i in range(n_users):
            idx = mask[i]  # Get indices of items observed by user i
            V_i = V[idx]  # Item factors for observed items
            R_i = R[i, idx]  # Ratings from user i for those items

            # Solve the regularized least squares system:
            A = V_i.T @ V_i + reg * I_f
            b = V_i.T @ R_i
            U[i] = np.linalg.solve(A, b)

        # Update item latent matrix V
        for j in range(n_items):
            idx = mask[:, j]  # Get indices of users who rated item j
            U_j = U[idx]  # User factors for those users
            R_j = R[idx, j]  # Ratings for item j from those users

            # Solve the regularized least squares system:
            A = U_j.T @ U_j + reg * I_f
            b = U_j.T @ R_j
            V[j] = np.linalg.solve(A, b)

    # Return the trained latent matrices
    return U, V

def cross_validate_als(df, cross_val_settings):
    """
    Performs cross-validation over a grid of ALS hyperparameters (number of latent factors and iterations),
    measuring model performance using Root Mean Squared Error (RMSE) on held-out test entries.

    Each fold is constructed by masking out a subset of observed entries, training the model on the rest,
    and then computing predictions for the masked entries. Mean imputation is used for missing values during training.

    args:
        df (DataFrame): Student–microgoal matrix (as returned by `load_student_data`), containing NaNs.
        cross_val_settings (dict): Dictionary containing cross-validation parameters:
            - "factors_list": list of latent factor counts
            - "iters_list": list of ALS iteration counts
            - "n_folds": number of cross-validation folds
            - "reg": regularization strength
            - "random_seed": random seed for reproducibility

    output:
        rmse_grid (ndarray): 2D array of average RMSE values with shape (len(factors_list), len(iters_list)).
    """

    # Unpack settings from the dictionary
    factors_list = cross_val_settings["factors_list"]
    iters_list = cross_val_settings["iters_list"]
    n_folds = cross_val_settings["n_folds"]
    reg = cross_val_settings["reg"]
    random_seed = cross_val_settings["random_seed"]

    # Convert DataFrame to NumPy array and identify non-missing values
    values = df.values.astype(float)
    mask_full = ~np.isnan(values)
    mu = values[mask_full].mean()
    coords = list(zip(*np.where(mask_full)))

    # Randomize
    np.random.seed(random_seed)
    np.random.shuffle(coords)
    folds = np.array_split(coords, n_folds)

    # Initialize RMSE grid to store average performance for each (factors, iterations) pair
    rmse_grid = np.zeros((len(factors_list), len(iters_list)))

    # Loop over all combinations of latent factors and iteration counts
    for i, n_factors in enumerate(factors_list):
        for j, n_iters in enumerate(iters_list):
            fold_rmses = []  # Track RMSEs for each fold

            # Loop over each cross-validation fold
            for k in range(len(folds)):

                # Create train/test masks for this fold
                mask_train = mask_full.copy()
                mask_test = np.zeros_like(mask_full, dtype=bool)

                for (u, v) in folds[k]:
                    mask_train[u, v] = False  # Remove test entries from training mask
                    mask_test[u, v] = True    # Add test entries to test mask

                # Replace missing values with global mean, center data by subtracting the mean
                R_train = np.where(mask_train, values, mu) - mu

                # Train ALS model on masked training data
                U, V = train_als(
                    R_train,
                    mask_train,
                    n_factors=n_factors,
                    n_iters=n_iters,
                    reg=reg
                )

                # === Predict the full matrix and re-add mean offset ===
                preds = U @ V.T + mu

                # === Evaluate performance on the test set ===
                diffs = preds[mask_test] - values[mask_test]
                mse = np.mean(diffs ** 2)
                rmse = np.sqrt(mse)
                fold_rmses.append(rmse)

                # === Print fold-level RMSE for monitoring ===
                print(f"Fold {k+1}/{len(folds)}, n_factors={n_factors}, n_iters={n_iters}, RMSE={rmse:.4f}")

            # === Average RMSE over all folds for this parameter combination ===
            rmse_grid[i, j] = np.mean(fold_rmses)

    # === Return the grid of RMSE values ===
    return rmse_grid

def get_best_params(rmse_grid, cross_val_settings):
    """
    Finds the best (lowest RMSE) parameter combination from the RMSE grid.

    args:
        rmse_grid (ndarray): 2D array of RMSE values.
        cross_val_settings (dict): Contains 'factors_list' and 'iters_list'.

    returns:
        best_f (int): Best number of latent factors.
        best_it (int): Best number of ALS iterations.
        best_rmse (float): Best RMSE value found.
        min_idx (tuple): Index of the best parameter combination.
    """
    factors_list = cross_val_settings["factors_list"]
    iters_list = cross_val_settings["iters_list"]

    min_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    best_f = factors_list[min_idx[0]]
    best_it = iters_list[min_idx[1]]
    best_rmse = rmse_grid[min_idx]

    return best_f, best_it, best_rmse, min_idx

def plot_rmse_heatmap(rmse_grid, cross_val_settings, title="CV: Root Mean Squared Error"):
    """
    Plots a heatmap of RMSE values from cross-validation over combinations
    of ALS hyperparameters (number of latent factors and iterations).

    args:
        rmse_grid (ndarray): 2D array of RMSE values with shape
                             [len(factors_list), len(iters_list)].
        cross_val_settings (dict): Dictionary containing hyperparameter lists:
            - "factors_list": list of latent factor counts (rows)
            - "iters_list": list of iteration counts (columns)
        title (str): Title of the plot (default: "CV: Root Mean Squared Error").

    output:
        Displays a heatmap of RMSE values.
    """
    # Extract parameter lists from settings
    factors_list = cross_val_settings["factors_list"]
    iters_list = cross_val_settings["iters_list"]

    # Create plot
    fig, ax = plt.subplots()
    cax = ax.imshow(rmse_grid, origin='lower', interpolation='nearest')

    # Set tick positions and labels
    ax.set_xticks(range(len(iters_list)))
    ax.set_xticklabels(iters_list)
    ax.set_yticks(range(len(factors_list)))
    ax.set_yticklabels(factors_list)

    # Set labels and title
    ax.set_xlabel('n_iters')
    ax.set_ylabel('n_factors')
    ax.set_title(title)

    # Add colorbar to show RMSE scale
    fig.colorbar(cax, ax=ax, label='RMSE')

    # Show plot
    plt.show()

if __name__ == '__main__':

    # Import the data
    df = pd.read_csv('../Data/shortened_student_abilities.csv', index_col=0)

    # Fetch the outcomes for the different training settings of Creating ALS
    rmse_grid = cross_validate_als(df, cross_val_settings)

    # Visualize the grid search
    plot_rmse_heatmap(rmse_grid,cross_val_settings)

    # Fetch the best parameters and export as JSON
    best_f, best_it, best_rmse, min_idx = get_best_params(rmse_grid, cross_val_settings)

    best_params = {
        "best_factors": int(best_f),
        "best_iters": int(best_it),
        "best_rmse": float(best_rmse),
        "best_index": [int(i) for i in min_idx]
    }

    with open('../Data/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)