# trading_incentive_optimization_app.py
# Title: Trading Incentive Optimization System
# Pages: 1) Load user data -> 2) Data table with manual override -> 3) Manual update (if enabled) -> 4) Summary view

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# ---------- CONFIG ----------
st.set_page_config(page_title="Trading Incentive Optimization System",
                   layout="wide",
                   page_icon="💹")

# Use relative filenames or full paths as you prefer
MODEL_PATH = "best_model.joblib"        # adjust if your model is at /mnt/data/best_model.joblib
TEST_PATH = "test_set_saved.csv"        # adjust if at /mnt/data/test_set_saved.csv
MAX_LOAD = 45
PAGE_SIZE = 20

# ---------- Ensure rerun compatibility ----------
if not hasattr(st, "rerun"):
    if hasattr(st, "experimental_rerun"):
        st.rerun = st.experimental_rerun
    else:
        def _noop():
            raise RuntimeError("Streamlit rerun not available in this version.")
        st.rerun = _noop

# ---------- TOP-LEVEL HELPERS (required for unpickling) ----------
# Define classes/functions here exactly as in training script so joblib.load can find them.

class RareLabelEncoder:
    """
    Keep only frequent categories per column (min_freq) and map others to '__OTHER__'.
    Must exist at top-level so joblib can unpickle the pipeline.
    """
    def __init__(self, min_freq=0.01, max_categories=None):
        self.min_freq = min_freq
        self.max_categories = max_categories
        self.frequent_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        n = len(X)
        for col in X.columns:
            vc = X[col].astype(str).value_counts(dropna=False)
            if self.min_freq is not None:
                allowed = list(vc[vc / n >= self.min_freq].index.astype(str))
            else:
                allowed = list(vc.index.astype(str))
            if self.max_categories is not None:
                allowed = allowed[:self.max_categories]
            self.frequent_[col] = set(allowed)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            allowed = self.frequent_.get(col, set())
            X[col] = X[col].astype(str).where(X[col].astype(str).isin(allowed), other="__OTHER__")
        return X

def to_str_func(X):
    """
    Convert input to DataFrame of strings (top-level function for pickling).
    """
    X_df = pd.DataFrame(X).copy()
    for c in X_df.columns:
        X_df[c] = X_df[c].where(X_df[c].notna(), other=np.nan)
        X_df[c] = X_df[c].astype(str)
    return X_df

# ---------- STYLES ----------
st.markdown("""
    <style>
        /* page layout padding so content isn't hidden behind the fixed footer */
        body, .block-container {
            padding-bottom: 90px;
        }

        .main-title {
            font-size:36px;
            color:#004aad;
            font-weight:700;
            text-align:center;
            margin-bottom:0.25em;
        }
        .sub-title {
            font-size:22px;
            color:#0085ff;
            font-weight:600;
            text-align:center;
            margin-bottom:1.5em;
        }
        .section-title {
            font-size:20px;
            font-weight:600;
            color:#004aad;
            margin-top:1.5em;
            margin-bottom:0.75em;
        }
        div.stButton > button {
            background-color:#004aad;
            color:white;
            border-radius:10px;
            height:3em;
            width:15em;
            font-size:16px;
            font-weight:600;
        }
        div.stButton > button:hover {
            background-color:#0073e6;
            color:white;
        }
        .footer-fixed {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(90deg, rgba(255,255,255,0.95), rgba(245,247,250,0.95));
            border-top: 1px solid #e6e9ef;
            padding: 10px 20px;
            text-align: center;
            color: #6b7280;
            font-size: 13px;
            z-index: 9999;
        }
        .small-caption { font-size:13px; color:gray; }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="main-title">Trading Incentive Optimization System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Admin Dashboard — Incentive Optimization Page</div>', unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "page" not in st.session_state:
    st.session_state.page = 1
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "display_data" not in st.session_state:
    st.session_state.display_data = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False
if "manual_table" not in st.session_state:
    st.session_state.manual_table = None

# ---------- Helper functions ----------
def detect_target_column(df):
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["grid","revenue","target","label","class"])]
    return candidates[0] if candidates else None

def load_test_data(path, max_rows=MAX_LOAD):
    df = pd.read_csv(path)
    if len(df) > max_rows:
        df = df.head(max_rows)
    target = detect_target_column(df)
    return df, target

def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Failed to load the model. See details below:")
        st.text(traceback.format_exc())
        st.stop()

def predict_df_with_model(model, df_display):
    X = df_display.copy()
    # Ensure same columns as training pipeline expected - pipeline should handle missing columns if designed that way
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.decision_function(X)
    preds = (probs >= 0.5).astype(int)
    # your requested mapping: 1->2, 0->1
    pred_grid = pd.Series(preds).map({1: 2, 0: 1})
    out = X.copy().reset_index(drop=True)
    out["Predicted_Probability"] = probs
    out["Predicted_Class_Binary"] = preds
    out["Predicted_Grid"] = pred_grid
    out["Eligibility_Status"] = np.where(pred_grid == 1, "Eligible (Grid 1)", "Not Eligible (Grid 2)")
    return out

# ---------------- Page 1: Load User Data ----------------
if st.session_state.page == 1:
    st.markdown("### Load Customer Data")
    st.info("Click below to load test user data and begin the incentive optimization process.")
    if st.button("📂 Load User Data"):
        if os.path.exists(TEST_PATH):
            raw_df, target_col = load_test_data(TEST_PATH, max_rows=MAX_LOAD)
            st.session_state.raw_data = raw_df.copy()
            # display_data excludes target/grid column
            if target_col:
                display_df = raw_df.drop(columns=[target_col])
            else:
                display_df = raw_df.copy()
            st.session_state.display_data = display_df.copy()
            st.session_state.page = 2
            st.rerun()
        else:
            st.error(f"Test data not found at {TEST_PATH}")

# ---------------- Page 2: Data Table & Predict ----------------
elif st.session_state.page == 2:
    df_display = st.session_state.display_data
    if df_display is None:
        st.warning("No data loaded. Please load data first.")
        if st.button("Back"):
            st.session_state.page = 1
            st.rerun()
    else:
        st.markdown('<div class="section-title">Loaded Customer Data</div>', unsafe_allow_html=True)
        st.dataframe(df_display.head(20), use_container_width=True, hide_index=True)
        st.caption("Showing 20 rows (max 45 entries loaded)")

        st.session_state.manual_mode = st.checkbox("Allow manual discount provision (Admin override)")
        if st.button("💡 Predict Eligible Users"):
            if os.path.exists(MODEL_PATH):
                model = safe_joblib_load(MODEL_PATH)
                df_out = predict_df_with_model(model, df_display)
                st.session_state.predictions = df_out.copy()
                # prepare editable table for manual mode (boolean column)
                st.session_state.manual_table = df_out[["Predicted_Grid"]].copy()
                # create boolean column 'Eligible' where True if Predicted_Grid == 1
                st.session_state.manual_table["Eligible"] = st.session_state.manual_table["Predicted_Grid"].apply(lambda x: True if x==1 else False)
                # navigate pages
                if st.session_state.manual_mode:
                    st.session_state.page = 3
                else:
                    st.session_state.page = 4
                st.rerun()
            else:
                st.error(f"Model not found at {MODEL_PATH}")

# ---------------- Page 3: Manual Mode (editable table) ----------------
elif st.session_state.page == 3:
    preds = st.session_state.predictions
    if preds is None:
        st.warning("No predictions available. Please run Predict first.")
        if st.button("Back to Data Page"):
            st.session_state.page = 2
            st.rerun()
    else:
        st.markdown('<div class="section-title">Manual Incentive Provision (Admin Override)</div>', unsafe_allow_html=True)
        st.caption("Edit the 'Eligible' column below and click 'Update Status' to save changes.")

        # Build an editable DataFrame for admin: include a few context columns plus Eligible boolean
        context_cols = preds.columns.tolist()
        # remove heavy columns if any
        editable_preview_cols = context_cols[:8] if len(context_cols) > 8 else context_cols
        # Build display DF: include index, a few context cols, Predicted_Grid and Eligible
        editable_df = preds.reset_index(drop=True)[editable_preview_cols].copy()
        # Ensure Predicted_Grid exists
        if "Predicted_Grid" not in editable_df.columns:
            editable_df["Predicted_Grid"] = preds["Predicted_Grid"].values
        # add Eligible boolean based on Predicted_Grid
        editable_df["Eligible"] = editable_df["Predicted_Grid"].apply(lambda x: True if x==1 else False)

        # Use data editor if available
        try:
            # new st.data_editor API
            edited = st.data_editor(editable_df, num_rows="fixed", use_container_width=True)
            # `edited` is the modified dataframe
            # apply edits: update st.session_state.predictions Predicted_Grid based on Eligible column
            if st.button("✅ Update Status"):
                # Map back: Eligible True -> Predicted_Grid = 1 else 2
                eligible_series = edited["Eligible"].astype(bool)
                updated_grid = eligible_series.map({True: 1, False: 2}).values
                # write back to session_state.predictions (match by row order)
                st.session_state.predictions = st.session_state.predictions.reset_index(drop=True)
                st.session_state.predictions["Predicted_Grid"] = updated_grid
                st.session_state.predictions["Predicted_Binary"] = (st.session_state.predictions["Predicted_Grid"] == 1).astype(int)
                st.session_state.predictions["Eligibility_Status"] = st.session_state.predictions["Predicted_Grid"].apply(lambda x: "Eligible (Grid 1)" if x==1 else "Not Eligible (Grid 2)")
                st.session_state.page = 4
                st.rerun()
        except Exception:
            # fallback: interactive checkboxes if data_editor not supported
            st.caption("Note: your Streamlit version does not support the inline data editor. Fallback to per-row checkboxes.")
            updated_checks = []
            for i, row in editable_df.iterrows():
                cols = st.columns([0.75, 0.25])
                with cols[0]:
                    st.markdown(f"**Row {i+1}** — Pred: Grid {int(row['Predicted_Grid'])} — Prob={preds.iloc[i]['Predicted_Probability']:.3f}")
                    preview_cols = [c for c in editable_df.columns if c not in ['Eligible','Predicted_Grid']]
                    preview = ", ".join([f"{c}: {str(row[c])}" for c in preview_cols[:4]])
                    st.caption(preview)
                with cols[1]:
                    checked = st.checkbox("Eligible", value=(row["Eligible"]==True), key=f"manual_chk_{i}")
                    updated_checks.append(checked)
            if st.button("✅ Update Status"):
                # write back
                updated_grid = [1 if chk else 2 for chk in updated_checks]
                st.session_state.predictions = st.session_state.predictions.reset_index(drop=True)
                st.session_state.predictions["Predicted_Grid"] = updated_grid
                st.session_state.predictions["Predicted_Binary"] = (st.session_state.predictions["Predicted_Grid"] == 1).astype(int)
                st.session_state.predictions["Eligibility_Status"] = st.session_state.predictions["Predicted_Grid"].apply(lambda x: "Eligible (Grid 1)" if x==1 else "Not Eligible (Grid 2)")
                st.session_state.page = 4
                st.rerun()

        if st.button("← Back to Data Page"):
            st.session_state.page = 2
            st.rerun()

# ---------------- Page 4: Results ----------------
elif st.session_state.page == 4:
    df = st.session_state.predictions.copy()
    if df is None:
        st.warning("No predictions available. Use the Data page to load and predict.")
        if st.button("Back to Data Page"):
            st.session_state.page = 2
            st.rerun()
    else:
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.success("Predictions complete. Select a view option below:")

        # KPI row
        total = len(df)
        eligible_count = int((df["Predicted_Grid"] == 1).sum())
        pct = eligible_count / total * 100 if total else 0
        k1, k2, k3 = st.columns(3)
        k1.metric("Total users", total)
        k2.metric("Eligible (Grid 1)", eligible_count)
        k3.metric("% Eligible", f"{pct:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            show_eligible = st.button("🎯 Show Eligible Users")
        with col2:
            show_all = st.button("👥 Show All Users")

        if show_eligible:
            eligible_df = df[df["Predicted_Grid"] == 1]
            st.markdown("### ✅ Eligible Users (Grid 1)")
            st.dataframe(eligible_df, use_container_width=True, hide_index=True)
            st.caption(f"{len(eligible_df)} out of {len(df)} users are eligible for trading incentive.")
            # download CSV button
            csv = eligible_df.to_csv(index=False)
            st.download_button("Download Eligible CSV", csv, "eligible_users.csv", "text/csv")
        elif show_all:
            st.markdown("### 📊 All Users with Eligibility Status")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Complete view of all users and their eligibility status.")
            csv = df.to_csv(index=False)
            st.download_button("Download All Predictions", csv, "all_users_predictions.csv", "text/csv")

        if st.button("Restart Process 🔄 "):
            st.session_state.page = 1
            st.session_state.raw_data = None
            st.session_state.display_data = None
            st.session_state.predictions = None
            st.session_state.manual_table = None
            st.rerun()

# ---------- Footer (fixed at bottom) ----------
st.markdown("""
    <div class="footer-fixed">
        © 2025 Trading Incentive Optimization System &nbsp;|&nbsp; Admin Dashboard — Incentive Optimization Page
    </div>
""", unsafe_allow_html=True)
