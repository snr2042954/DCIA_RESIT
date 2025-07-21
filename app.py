import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Add BackendPipeline to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "BackendPipeline"))

# For a better description of how these functions work please look at README_backend or in the respective .py files
from ALS_grid_search import train_als
from ALS_recommendations import (
    get_completed_microgoals,
    recommend_microgoals_within_completed_islands,
    load_microgoal_features
)

### LOADING THE DATA
@st.cache_data
def load_data():
    """
    Loads and prepares the student–microgoal dataset and trained ALS model parameters.

    This function performs the following steps:
    1. Loads the student ability matrix and microgoal metadata.
    2. Cleans the microgoal metadata and extracts human-readable names.
    3. Computes the global mean (mu) used to center the data.
    4. Loads the best ALS hyperparameters from JSON (latent factors and iterations).
    5. Filters the dataset to include only microgoals present in the metadata.
    6. Trains the ALS model on the full dataset using the best parameters.

    Returns:
        df_students (pd.DataFrame): Cleaned student–microgoal ability matrix.
        df_hierarchy (pd.DataFrame): Microgoal metadata including world/island structure.
        microgoal_names (pd.Series): Mapping from microgoal_id to readable names.
        U (np.ndarray): Trained ALS user latent matrix.
        V (np.ndarray): Trained ALS item (microgoal) latent matrix.
        mu (float): Global mean ELO score used for centering the matrix.
        best_params (dict): Best ALS hyperparameters loaded from JSON.
    """
    df_students = pd.read_csv('Data/shortened_student_abilities.csv', index_col=0)
    df_hierarchy = load_microgoal_features('Data/microgoal_features.csv')
    microgoal_names = df_hierarchy.reset_index().set_index('microgoal_id')['name']

    # Prepare for ALS
    values = df_students.values.astype(float)
    mask_full = ~np.isnan(values)
    mu = values[mask_full].mean()

    with open('Data/best_params.json', 'r') as f:
        best_params = json.load(f)

    best_f = best_params["best_factors"]
    best_it = best_params["best_iters"]

    # Filter for consistent microgoals
    common_ids = set(df_students.columns.astype(str)).intersection(microgoal_names.index.astype(str))
    df_students = df_students.loc[:, df_students.columns.intersection(common_ids)]
    microgoal_names = microgoal_names.loc[list(common_ids)]

    R_train = np.where(mask_full, values, mu) - mu
    U, V = train_als(R_train, mask_full, n_factors=best_f, n_iters=best_it, reg=0.1)

    return df_students, df_hierarchy, microgoal_names, U, V, mu, best_params

### SETTING UP THE UI

st.set_page_config(page_title="SafeSkip", page_icon="assets/SafeSkip.png", layout="centered")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "SafeSkip Recommender"])

if page == "Home":

    st.image("assets/SafeSkip.png")

    st.title("SafeSkip Recommender System")

    st.markdown("""
    
    **SafeSkip** is an intelligent recommendation tool built by a student consultancy team at JADS in collaboration with Gynzy.  
    It is designed to identify which educational *microgoals* a student can **safely skip**, allowing for a more efficient, personalized learning experience without compromising mastery.
    
    The system uses **ALS (Alternating Least Squares)** matrix factorization to model each student's ability based on historical data. It then recommends only those microgoals:
    - That the student hasn’t completed yet,
    - That belong to topics (islands) the student has already engaged with,
    - And for which their predicted performance is confidently above the mastery threshold.
    
    This interface lets you explore which microgoals a student has already mastered, and which ones can potentially be skipped backed by real model predictions and evaluation metrics.
    """, unsafe_allow_html=True)

if page == "SafeSkip Recommender":

    st.header("Data Selection")

    # Load in the data from the data load function
    df_students, df_hierarchy, microgoal_names, U, V, mu, best_params = load_data()
    student_ids = df_students.index.tolist()
    selected_user = st.selectbox("Select a student", student_ids)

    threshold = st.slider("ELO Pass Threshold", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    st.caption("By selecting a threshold you decide from which ELO score onwards a micro-goal is considered to have been passed. "
               "This concerns already completed microgoals AND the predictions. "
               "The threshold therefore updates the completed microgoals, the SafeSkip recommendatons and the evaluation metrics below.")

    top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
    st.caption("This slider simply limits the amount of recommendations that the SafeSkip system shows.")

    st.markdown("---")

    st.header("Microgoal Overview")

    # This try loop contains all the actual lookups from the data which we just selected
    # It also makes the predictions
    try:

        # Get completed Microgoals from set threshold
        completed = get_completed_microgoals(selected_user, df_students, microgoal_names, pass_threshold=threshold)
        st.subheader("This Student's Completed Micro-goals")
        if completed:
            st.dataframe(pd.DataFrame(completed, columns=["ID", "Name", "ELO Score"]))
        else:
            st.info("This student has not completed any microgoals above the threshold.")

        # Get the SafeSkip Recommendations
        recs = recommend_microgoals_within_completed_islands(
            selected_user, df_students, U, V, mu,
            microgoal_names, df_hierarchy,
            threshold=threshold, top_n=top_n
        )

        # Fetch safeskip recommendations and put into table
        st.subheader("SafeSkip Recommendations")
        if recs:
            st.dataframe(pd.DataFrame(recs, columns=["ID", "Name", "Predicted ELO"]))
        else:
            st.warning("No recommended microgoals found for this student in completed islands.")

        # Show top recommendations in a bar graph with the threshold line
        st.subheader("Predicted Mastery (Bar Chart of Recommendations)")
        if recs:
            rec_df = pd.DataFrame(recs, columns=["ID", "Name", "Predicted ELO"])
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            sns.barplot(data=rec_df, x="Name", y="Predicted ELO", ax=ax_bar)

            # insert threshold bar
            ax_bar.axhline(threshold, color='red', linestyle='--', label='Threshold')

            # Shorten long x-tick labels to only 20 characters
            labels = [label.get_text() for label in ax_bar.get_xticklabels()]
            short_labels = [label[:20] + '...' if len(label) > 20 else label for label in labels]
            ax_bar.set_xticklabels(short_labels, rotation=45, ha="right")

            # Set title and legend and plot
            ax_bar.set_title("Top Predicted Microgoals (Safe to Skip)")
            ax_bar.legend()
            st.pyplot(fig_bar)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.header("Model Evaluation")

    # Initiate a button whereafter the evaluations are shown
    if st.button("Run Evaluation on Model"):
        values = df_students.values.astype(float)
        mask_full = ~np.isnan(values)
        mu = values[mask_full].mean()
        coords = list(zip(*np.where(mask_full)))

        # Create test/train split (20% hidden)
        mask_demo = mask_full.copy()
        mask_test_demo = np.zeros_like(mask_full, dtype=bool)
        np.random.seed(42)
        np.random.shuffle(coords)
        n_hide_demo = int(len(coords) * 0.2)
        for (u, v) in coords[:n_hide_demo]:
            mask_demo[u, v] = False
            mask_test_demo[u, v] = True

        # Train ALS on training portion
        R_train_demo = np.where(mask_demo, values, mu) - mu
        U_demo, V_demo = train_als(
            R_train_demo,
            mask_demo,
            n_factors=best_params["best_factors"],
            n_iters=best_params["best_iters"],
            reg=0.1
        )
        preds_demo = U_demo @ V_demo.T + mu

        # Evaluate (original function uses plt which is not supported
        # Compute binary classification labels
        true_bin = (values[mask_test_demo] > threshold).astype(int)
        pred_bin = (preds_demo[mask_test_demo] > threshold).astype(int)

        st.subheader("Evaluation Metrics")

        # Compute metrics
        accuracy = accuracy_score(true_bin, pred_bin)
        precision = precision_score(true_bin, pred_bin)
        recall = recall_score(true_bin, pred_bin)
        f1 = f1_score(true_bin, pred_bin)

        # Display metrics
        st.markdown(f"""
        **Threshold = {threshold}**
    
        - Accuracy: `{accuracy:.2f}`
        - Precision: `{precision:.2f}`
        - Recall: `{recall:.2f}`
        - F1 Score: `{f1:.2f}`
        """)

        st.subheader("Confusion Matrix")

        # Plot confusion matrix
        cm = confusion_matrix(true_bin, pred_bin)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Initiate explanatory tabs
        tab1, tab2, tab3, tab4 = st.tabs(["TP", "TN", "FP", "FN"])

        # Unpack confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        # Fill explanatory tabs
        with tab1:
            st.metric(label="True Positives (TP):", value=tp)
            st.write("These are cases where the model **correctly predicted mastery** (ELO ≥ threshold)."
                     "(Bottom right)")

        with tab2:
            st.metric(label="True Negatives (TN):", value=tn)
            st.write("The model **correctly predicted lack of mastery** (ELO < threshold)."
                     "(Top left)")

        with tab3:
            st.metric(label="False Positives (FP):", value=fp)
            st.write("Model **falsely predicted mastery** when the student had not actually mastered the microgoal."
                     "(Top Right)")

        with tab4:
            st.metric(label="False Negatives (FN):", value=fn)
            st.write("Model **missed actual mastery** — predicted below threshold, but the student had mastered it."
                     "(Bottom Left)")
