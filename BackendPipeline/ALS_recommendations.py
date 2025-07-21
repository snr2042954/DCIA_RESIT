import pandas as pd
import numpy as np
from ALS_grid_search import train_als
import json

def get_completed_microgoals(user_id, df, microgoal_names, pass_threshold=1.0):
    """
    Identify microgoals that a student has completed based on a given threshold.

    Input:
        user_id (str): The ID of the student to evaluate
        df (DataFrame): Student–microgoal ability matrix (ELO scores)
        microgoal_names (Series): Mapping from microgoal_id to readable name
        pass_threshold (float): Minimum ELO score to count as 'completed'

    Output:
        List of tuples: [(microgoal_id, name, ELO score), ...]
    """

    if user_id not in df.index:
        raise ValueError(f"User {user_id} not found in data.")

    # Extract row for the given user
    user_row = df.loc[user_id]

    # Filter for completed microgoals (above threshold)
    completed = user_row[user_row >= pass_threshold].dropna()

    # Map IDs to names and values
    ids = completed.index.tolist()
    scores = completed.values.tolist()
    names = [microgoal_names.get(str(mid), '<unknown>') for mid in ids]

    return list(zip(ids, names, scores))

def recommend_microgoals_within_completed_islands(user_id, df, U, V, mu, microgoal_names, df_hierarchy,
                                                  threshold=1.0, top_n=10):
    """
    Recommend unseen microgoals that belong to islands the student has already engaged with.

    Input:
        user_id (str): ID of the target student
        df (DataFrame): Student–microgoal matrix (ELO scores)
        U, V (ndarrays): Trained ALS latent matrices for users and items
        mu (float): Global mean used for centering
        microgoal_names (Series): Microgoal name lookup
        df_hierarchy (DataFrame): Metadata including world/island/microgoal_id
        threshold (float): Minimum predicted ELO to consider for recommendation
        top_n (int): Maximum number of recommendations to return

    Output:
        List of tuples: [(microgoal_id, name, predicted ELO), ...]
    """

    # Get IDs of completed microgoals for this student
    completed = get_completed_microgoals(user_id, df, microgoal_names, threshold)
    completed_ids = set(str(mid) for mid, _, _ in completed)

    # Determine which islands the user has completed something in
    completed_islands = set()
    for idx, row in df_hierarchy.iterrows():
        if row['microgoal_id'] in completed_ids:
            completed_islands.add(idx)

    # Gather all microgoals from those completed islands
    allowed_microgoal_ids = set()
    for idx in completed_islands:
        allowed_microgoal_ids.update(
            df_hierarchy.loc[[idx], 'microgoal_id'].values
        )

    # Predict scores for all microgoals
    idx_user = df.index.get_loc(user_id)
    raw_preds = U[idx_user] @ V.T
    preds = raw_preds + mu  # Recenter predictions

    # Identify microgoals that haven't been attempted yet
    attempted = ~np.isnan(df.values[idx_user])
    unseen = np.where(~attempted)[0]
    unseen_ids = df.columns[unseen].astype(str)

    # Filter to unseen microgoals in allowed islands with predicted score ≥ threshold
    candidates = [
        i for i, mid in zip(unseen, unseen_ids)
        if mid in allowed_microgoal_ids and preds[i] >= threshold
    ]
    ranked = sorted(candidates, key=lambda i: preds[i], reverse=True)
    top_idxs = ranked[:top_n]

    # Convert indices to readable output
    ids = df.columns[top_idxs].tolist()
    scores = [preds[i] for i in top_idxs]
    names = [microgoal_names.get(str(mid), '<unknown>') for mid in ids]

    return list(zip(ids, names, scores))

def load_microgoal_features(filepath):
    """
    Load microgoal metadata with names, worlds, and islands.

    Input:
        filepath (str): Path to CSV file with columns: microgoal_id, name, world, island

    Output:
        DataFrame: Indexed by (world, island), with microgoal metadata
    """
    mf = pd.read_csv(filepath, dtype={'microgoal_id': str})

    # Ensure the column is named correctly
    if 'microgoal_id' not in mf.columns:
        mf = mf.rename(columns={mf.columns[0]: 'microgoal_id'})

    mf['microgoal_id'] = mf['microgoal_id'].astype(str)

    # Return indexed by (world, island)
    mf = mf[['microgoal_id', 'name', 'world', 'island']].set_index(['world', 'island'], drop=True)
    mf.sort_index(inplace=True)

    return mf


if __name__ == '__main__':

    verbose = True

    # Load the data (also microgoal data this time)
    df_students = pd.read_csv('../Data/shortened_student_abilities.csv', index_col=0)
    df_hierarchy = load_microgoal_features('../Data/microgoal_features.csv')
    microgoal_names = df_hierarchy.set_index('microgoal_id')['name']

    # Convert DataFrame to NumPy array and identify non-missing values
    values = df_students.values.astype(float)
    mask_full = ~np.isnan(values)
    mu = values[mask_full].mean()
    coords = list(zip(*np.where(mask_full)))

    # Load the best parameters from the JSON file
    with open('../Data/best_params.json', 'r') as f:
        best_params = json.load(f)
    best_f = best_params["best_factors"]  # int: best number of latent factors
    best_it = best_params["best_iters"]  # int: best number of ALS iterations
    best_rmse = best_params["best_rmse"]  # float: best RMSE value
    min_idx = tuple(best_params["best_index"])  # tuple: index in RMSE grid

    # Filter columns to only those microgoals present in metadata
    common_ids = set(df_students.columns.astype(str)).intersection(microgoal_names.index.astype(str))
    df_students = df_students.loc[:, df_students.columns.intersection(common_ids)]
    microgoal_names = microgoal_names.loc[list(common_ids)]

    # Prepare matrix and train ALS on best parameters
    R_train = np.where(mask_full, values, mu) - mu
    U, V = train_als(R_train, mask_full, n_factors=best_f, n_iters=best_it, reg=0.1)

    # Select an example user and limit the nr of microgoals shown. also thoose top_n for later
    example_user = df_students.index[1]
    nr_microgoals = 20
    top_n = 5

    # Fetch and shorten completed microgoals
    completed = get_completed_microgoals(example_user, df_students, microgoal_names)
    completed_shortened = completed[:nr_microgoals]

    if verbose:
        print(f"\nCompleted microgoals (first {nr_microgoals}) for {example_user} (ELO score):")
        for mid, name, score in completed_shortened:
            print(f"  {mid}: {name} (ELO: {score:.2f})")

    # Generate recommendations
    recs = recommend_microgoals_within_completed_islands(example_user, df_students, U, V, mu, microgoal_names,
                                                         df_hierarchy, threshold=1.0, top_n=top_n)

    # Print recommendations
    print(f"\nTop {top_n} recommendations for {example_user} (predicted ELO):")
    for mid, name, score in recs[:top_n]:
        print(f"  {mid}: {name} (predicted ELO: {score:.2f})")