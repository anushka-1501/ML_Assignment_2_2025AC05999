
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Classifier",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Employee Attrition Classification")

st.markdown(
    """
    ### Machine Learning Model Comparison

    This application demonstrates employee attrition classification
    using five different machine learning models.

    **Models implemented:**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Gaussian Naive Bayes
    - Random Forest
    """
)

st.divider()

# ============================================================
# LOAD SAVED MODELS
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if os.path.exists(model_path):
            loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


models = load_models()

# ============================================================
# CHECK MODELS
# ============================================================

if len(models) == 0:

    st.error(
        "No trained models were found. "
        "Please make sure the model folder contains the .pkl files."
    )

    st.stop()

# ============================================================
# MODEL SELECTION
# ============================================================

st.subheader("1️⃣ Select a Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.success(
    f"Selected Model: **{selected_model_name}**"
)

st.divider()

# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("2️⃣ Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        test_data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            test_data.head(10),
            use_container_width=True
        )

        st.write(
            f"**Rows:** {test_data.shape[0]}  |  "
            f"**Columns:** {test_data.shape[1]}"
        )

        # ----------------------------------------------------
        # Check target
        # ----------------------------------------------------

        if "Attrition" not in test_data.columns:

            st.error(
                "The uploaded CSV must contain the "
                "'Attrition' column for evaluation."
            )

            st.stop()

        # ----------------------------------------------------
        # Separate features and target
        # ----------------------------------------------------

        X_test_app = test_data.drop(
            "Attrition",
            axis=1
        )

        y_test_app = test_data["Attrition"]

        # Handle target if it is stored as Yes/No
        if y_test_app.dtype == "object":

            y_test_app = y_test_app.map({
                "No": 0,
                "Yes": 1
            })

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = selected_model.predict(
            X_test_app
        )

        # ----------------------------------------------------
        # Probability for AUC
        # ----------------------------------------------------

        if hasattr(
            selected_model,
            "predict_proba"
        ):

            y_prob = selected_model.predict_proba(
                X_test_app
            )[:, 1]

        else:

            y_prob = selected_model.decision_function(
                X_test_app
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test_app,
            y_pred
        )

        auc = roc_auc_score(
            y_test_app,
            y_prob
        )

        precision = precision_score(
            y_test_app,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test_app,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test_app,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test_app,
            y_pred
        )

        # ====================================================
        # DISPLAY METRICS
        # ====================================================

        st.subheader("3️⃣ Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

        with col2:

            st.metric(
                "AUC",
                f"{auc:.4f}"
            )

        with col3:

            st.metric(
                "Precision",
                f"{precision:.4f}"
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            st.metric(
                "Recall",
                f"{recall:.4f}"
            )

        with col5:

            st.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

        with col6:

            st.metric(
                "MCC",
                f"{mcc:.4f}"
            )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.divider()

        st.subheader("4️⃣ Confusion Matrix")

        cm = confusion_matrix(
            y_test_app,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual: No Attrition",
                "Actual: Attrition"
            ],
            columns=[
                "Predicted: No Attrition",
                "Predicted: Attrition"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader(
            "5️⃣ Classification Report"
        )

        report = classification_report(
            y_test_app,
            y_pred,
            target_names=[
                "No Attrition",
                "Attrition"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ====================================================
        # PREDICTIONS
        # ====================================================

        st.subheader(
            "6️⃣ Prediction Results"
        )

        prediction_results = test_data.copy()

        prediction_results[
            "Predicted Attrition"
        ] = np.where(
            y_pred == 1,
            "Yes",
            "No"
        )

        st.dataframe(
            prediction_results,
            use_container_width=True
        )

        # ====================================================
        # DOWNLOAD PREDICTIONS
        # ====================================================

        csv_output = prediction_results.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Predictions",
            data=csv_output,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            "An error occurred while processing the dataset."
        )

        st.exception(e)

else:

    st.info(
        "Please upload the test_data.csv file to "
        "evaluate the selected model."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Assignment 2 | "
    "Employee Attrition Classification"
)
