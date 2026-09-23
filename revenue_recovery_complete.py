"""
revenue_recovery_complete.py
============================
Revenue Recovery Agent — ALL-IN-ONE SINGLE FILE
IBM SkillsBuild / BharatCares Internship Project

Contains ALL modules in one file for easy submission:
  - Data Preprocessing
  - EDA / Visualisations
  - Feature Engineering
  - Model Training & Evaluation
  - Recovery Prediction Engine
  - Recovery Agent (Priority Scoring + Action Recommendations)
  - Database Utilities
  - Streamlit Dashboard (all 6 pages)

Run:
    streamlit run revenue_recovery_complete.py

Dataset:
    Place Case_Data.csv in a data/ subfolder OR in the same directory.
"""

import os
import sys
import sqlite3
import textwrap
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CLASSIFIER_PATH = MODELS_DIR / "recovery_classifier.pkl"
REGRESSOR_PATH  = MODELS_DIR / "recovery_regressor.pkl"
LABEL_ENC_PATH  = MODELS_DIR / "label_encoder.pkl"
FEAT_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
DB_PATH         = DATA_DIR / "recovery_agent.db"

# ═══════════════════════════════════════════════════════════════════════════
# ENCODING MAPS
# ═══════════════════════════════════════════════════════════════════════════

GRADE_MAP = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
HOME_MAP  = {"OWN":0,"RENT":1,"MORTGAGE":2,"OTHER":3,"NONE":4,"ANY":5}
VERIF_MAP = {"Not Verified":0,"Source Verified":1,"Verified":2}
EMP_MAP   = {"< 1 year":0,"1 year":1,"2 years":2,"3 years":3,"4 years":4,
             "5 years":5,"6 years":6,"7 years":7,"8 years":8,"9 years":9,"10+ years":10}

FEATURE_COLS = [
    "loan_amnt","funded_amnt","int_rate","installment","annual_inc","dti",
    "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal","revol_util",
    "total_acc","total_pymnt","total_rec_prncp","total_rec_int","total_rec_late_fee",
    "last_pymnt_amnt","collections_12_mths_ex_med","acc_now_delinq","tot_coll_amt",
    "tot_cur_bal","total_rev_hi_lim","term_months","emp_length_years","grade_encoded",
    "home_ownership_encoded","verification_status_encoded","purpose_encoded",
    "months_outstanding","high_dti_flag","high_int_rate_flag","no_delinq_flag",
    "payment_completeness","income_to_loan_ratio","revol_pressure",
]

PRIORITY_COLOURS  = {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}
RECOVERY_COLOURS  = {"High Recovery":"#22c55e","Medium Recovery":"#f59e0b","Low Recovery":"#ef4444"}
PRIORITY_EMOJI    = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}

# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 1: DATA PREPROCESSING ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _find_data_file() -> Path:
    for candidate in [DATA_DIR/"Case_Data.csv", PROJECT_ROOT/"Case_Data.csv"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Case_Data.csv not found. Place it in the data/ folder next to this script.")


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(_find_data_file(), low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.loc[:, df.columns.str.strip() != ""]
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace({"nan": np.nan, "": np.nan}, inplace=True)
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.replace("%","").astype(float)
    else:
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    num_cols = ["loan_amnt","funded_amnt","installment","annual_inc","dti",
                "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal",
                "revol_util","total_acc","out_prncp","total_pymnt","total_rec_prncp",
                "total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee",
                "last_pymnt_amnt","collections_12_mths_ex_med","mths_since_last_delinq",
                "mths_since_last_major_derog","acc_now_delinq","tot_coll_amt",
                "tot_cur_bal","total_rev_hi_lim","annual_inc_joint","dti_joint"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else "Unknown")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["term_months"]                = df["term"].astype(float)
    df["emp_length_years"]           = df["emp_length"].map(EMP_MAP).fillna(5.0)
    df["grade_encoded"]              = df["grade"].map(GRADE_MAP).fillna(4.0)
    df["home_ownership_encoded"]     = df["home_ownership"].map(HOME_MAP).fillna(3.0)
    df["verification_status_encoded"]= df["verification_status"].map(VERIF_MAP).fillna(1.0)
    purposes = sorted(df["purpose"].dropna().unique())
    df["purpose_encoded"]            = df["purpose"].map({p:i for i,p in enumerate(purposes)}).fillna(0)
    df["recovery_ratio"]             = (df["total_pymnt"]/df["loan_amnt"]).clip(0,1)
    df["payment_ratio"]              = (df["total_rec_prncp"]/df["loan_amnt"]).clip(0,1)
    df["outstanding_amount"]         = (df["loan_amnt"]-df["total_rec_prncp"]).clip(lower=0)
    safe_inst = df["installment"].replace(0, np.nan)
    df["months_outstanding"]         = ((df["loan_amnt"]-df["total_pymnt"])/safe_inst).clip(lower=0).fillna(0)
    def cls(r):
        return "High Recovery" if r>=0.60 else ("Medium Recovery" if r>=0.20 else "Low Recovery")
    df["recovery_class"]             = df["recovery_ratio"].apply(cls)
    df["recovery_binary"]            = (df["recovery_ratio"]>=0.20).astype(int)
    df["high_dti_flag"]              = (df["dti"]>35).astype(int)
    df["high_int_rate_flag"]         = (df["int_rate"]>20).astype(int)
    df["no_delinq_flag"]             = (df["delinq_2yrs"]==0).astype(int)
    exp_total = df["installment"]*df["term_months"]
    df["payment_completeness"]       = (df["total_pymnt"]/exp_total.replace(0,np.nan)).fillna(0).clip(0,1)
    df["income_to_loan_ratio"]       = (df["annual_inc"]/df["loan_amnt"].replace(0,np.nan)).fillna(1).clip(0,50)
    df["revol_pressure"]             = (df["revol_bal"]/df["total_rev_hi_lim"].replace(0,np.nan)).fillna(0).clip(0,2)
    return df


def load_and_preprocess() -> pd.DataFrame:
    return engineer_features(clean_data(load_raw_data()))


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 2: EDA / KPI FUNCTIONS ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> dict:
    tot_loan    = df["loan_amnt"].sum()
    tot_recov   = df["total_pymnt"].sum()
    tot_princ   = df["total_rec_prncp"].sum()
    tot_out     = df["outstanding_amount"].sum() if "outstanding_amount" in df else (df["loan_amnt"]-df["total_rec_prncp"]).clip(0).sum()
    rec_rate    = (tot_princ/tot_loan*100) if tot_loan>0 else 0
    return {
        "Total Accounts":                  len(df),
        "Total Loan Amount (₹)":           round(tot_loan,2),
        "Total Amount Recovered (₹)":      round(tot_recov,2),
        "Principal Recovered (₹)":         round(tot_princ,2),
        "Total Outstanding Amount (₹)":    round(tot_out,2),
        "Recovery Rate (%)":               round(rec_rate,2),
        "High Risk Accounts":              int((df["recovery_class"]=="Low Recovery").sum()) if "recovery_class" in df else 0,
        "Medium Risk Accounts":            int((df["recovery_class"]=="Medium Recovery").sum()) if "recovery_class" in df else 0,
        "High Recovery Accounts":          int((df["recovery_class"]=="High Recovery").sum()) if "recovery_class" in df else 0,
        "Avg Loan Amount (₹)":             round(df["loan_amnt"].mean(),2),
        "Avg Recovered Amount (₹)":        round(df["total_pymnt"].mean(),2),
        "Avg Outstanding Amount (₹)":      round(tot_out/len(df),2) if len(df)>0 else 0,
        "Total Late Fees Recovered (₹)":   round(df["total_rec_late_fee"].sum(),2) if "total_rec_late_fee" in df else 0,
        "Total Interest Recovered (₹)":    round(df["total_rec_int"].sum(),2) if "total_rec_int" in df else 0,
        "Avg Interest Rate (%)":           round(df["int_rate"].mean(),2),
        "Avg Debt-to-Income Ratio":        round(df["dti"].mean(),2),
    }

def _cmap():
    return {"High Recovery":"#22c55e","Medium Recovery":"#f59e0b","Low Recovery":"#ef4444"}

def plot_recovery_class_dist(df):
    c=df["recovery_class"].value_counts().reset_index(); c.columns=["Class","Count"]
    return px.pie(c,names="Class",values="Count",title="Recovery Class Distribution",
                  color="Class",color_discrete_map=_cmap(),hole=0.4).update_layout(template="plotly_white",height=380)

def plot_loan_dist(df):
    return px.histogram(df,x="loan_amnt",nbins=30,title="Loan Amount Distribution",
                        labels={"loan_amnt":"Loan Amount (₹)"},color_discrete_sequence=["#3b82f6"]
                        ).update_layout(template="plotly_white",height=380)

def plot_recovery_ratio_dist(df):
    fig=px.histogram(df,x="recovery_ratio",nbins=25,title="Recovery Ratio Distribution",
                     color_discrete_sequence=["#8b5cf6"]).update_layout(template="plotly_white",height=380)
    fig.add_vline(x=0.20,line_dash="dash",line_color="orange",annotation_text="0.20 threshold")
    fig.add_vline(x=0.60,line_dash="dash",line_color="green",annotation_text="0.60 threshold")
    return fig

def plot_outstanding_grade(df):
    g=df.groupby("grade")["outstanding_amount"].mean().reset_index()
    return px.bar(g,x="grade",y="outstanding_amount",title="Avg Outstanding by Grade",
                  color="outstanding_amount",color_continuous_scale="Reds",text_auto=".0f"
                  ).update_layout(template="plotly_white",height=380)

def plot_recovery_grade(df):
    g=df.groupby("grade")["recovery_ratio"].mean().reset_index()
    return px.bar(g,x="grade",y="recovery_ratio",title="Avg Recovery Ratio by Grade",
                  color="recovery_ratio",color_continuous_scale="Greens",text_auto=".2f"
                  ).update_layout(template="plotly_white",height=380)

def plot_recovery_purpose(df):
    g=df.groupby("purpose")["recovery_ratio"].mean().reset_index().sort_values("recovery_ratio")
    return px.bar(g,x="recovery_ratio",y="purpose",orientation="h",
                  title="Avg Recovery Ratio by Purpose",
                  color="recovery_ratio",color_continuous_scale="Blues",text_auto=".2f"
                  ).update_layout(template="plotly_white",height=450)

def plot_loan_vs_recovered(df):
    fig=px.scatter(df,x="loan_amnt",y="total_pymnt",color="recovery_class",
                   color_discrete_map=_cmap(),title="Loan Amount vs Total Recovered",
                   labels={"loan_amnt":"Loan Amount","total_pymnt":"Total Payment"},opacity=0.7)
    mx=df["loan_amnt"].max()
    fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",name="Full Recovery",
                             line=dict(dash="dash",color="gray")))
    return fig.update_layout(template="plotly_white",height=420)

def plot_int_rate_box(df):
    return px.box(df,x="recovery_class",y="int_rate",color="recovery_class",
                  color_discrete_map=_cmap(),title="Interest Rate by Recovery Class"
                  ).update_layout(template="plotly_white",height=380,showlegend=False)

def plot_dti_box(df):
    return px.box(df,x="recovery_class",y="dti",color="recovery_class",
                  color_discrete_map=_cmap(),title="DTI by Recovery Class"
                  ).update_layout(template="plotly_white",height=380,showlegend=False)

def plot_home_ownership(df):
    g=df.groupby(["home_ownership","recovery_class"]).size().reset_index(name="Count")
    return px.bar(g,x="home_ownership",y="Count",color="recovery_class",barmode="group",
                  color_discrete_map=_cmap(),title="Recovery Class by Home Ownership"
                  ).update_layout(template="plotly_white",height=380)

def plot_term_recovery(df):
    d=df.copy(); d["term_lbl"]=(d["term"].astype(int)).astype(str)+" months"
    g=d.groupby("term_lbl")["recovery_ratio"].mean().reset_index()
    return px.bar(g,x="term_lbl",y="recovery_ratio",title="Recovery by Term",
                  color="recovery_ratio",color_continuous_scale="Teal",text_auto=".3f"
                  ).update_layout(template="plotly_white",height=350)

def plot_corr_heatmap(df):
    cols=["loan_amnt","int_rate","installment","annual_inc","dti","delinq_2yrs",
          "revol_bal","revol_util","total_acc","total_pymnt","total_rec_prncp",
          "total_rec_int","recovery_ratio","outstanding_amount"]
    avail=[c for c in cols if c in df.columns]
    return px.imshow(df[avail].corr(),text_auto=".2f",title="Correlation Heatmap",
                     color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto"
                     ).update_layout(template="plotly_white",height=550)

def plot_income_scatter(df):
    return px.scatter(df,x="annual_inc",y="recovery_ratio",color="recovery_class",
                      color_discrete_map=_cmap(),title="Annual Income vs Recovery Ratio",
                      log_x=True,opacity=0.65).update_layout(template="plotly_white",height=400)

def plot_top_risk(df,n=15):
    id_col="id" if "id" in df.columns else df.columns[0]
    top=df.nlargest(n,"outstanding_amount")[[id_col,"outstanding_amount","recovery_class"]].copy()
    top["Acct"]=top[id_col].astype(str).str[-6:]
    return px.bar(top,x="Acct",y="outstanding_amount",color="recovery_class",
                  color_discrete_map=_cmap(),title=f"Top {n} by Outstanding Amount",
                  text_auto=".0f").update_layout(template="plotly_white",height=420)

def plot_gauge(rate):
    return go.Figure(go.Indicator(mode="gauge+number+delta",value=rate,
        delta={"reference":50},title={"text":"Recovery Rate (%)"},
        gauge={"axis":{"range":[0,100]},"bar":{"color":"#3b82f6"},
               "steps":[{"range":[0,30],"color":"#fecaca"},
                        {"range":[30,60],"color":"#fef3c7"},
                        {"range":[60,100],"color":"#d1fae5"}],
               "threshold":{"line":{"color":"red","width":4},"thickness":0.75,"value":50}}
    )).update_layout(height=300)

def plot_verif_recovery(df):
    g=df.groupby(["verification_status","recovery_class"]).size().reset_index(name="Count")
    return px.bar(g,x="verification_status",y="Count",color="recovery_class",barmode="stack",
                  color_discrete_map=_cmap(),title="Recovery by Verification Status"
                  ).update_layout(template="plotly_white",height=380)

def plot_confusion(cm,labels):
    return px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Blues",
                     labels=dict(x="Predicted",y="Actual"),title="Confusion Matrix"
                     ).update_layout(template="plotly_white",height=400)

def plot_feat_imp(imp,n=15):
    top=imp.head(n).sort_values()
    fig=go.Figure(go.Bar(x=top.values,y=top.index,orientation="h",marker_color="#3b82f6"))
    return fig.update_layout(title=f"Top {n} Feature Importances",
                             xaxis_title="Importance",template="plotly_white",height=450)

def plot_pred_scatter(y_pred,y_test):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=y_test,y=y_pred,mode="markers",
                             marker=dict(color="#3b82f6",opacity=0.6),name="Predictions"))
    mx=max(float(np.max(y_test)),float(np.max(y_pred)),0.01)
    fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",
                             line=dict(dash="dash",color="red"),name="Perfect"))
    return fig.update_layout(title="Actual vs Predicted Recovery Ratio",
                             xaxis_title="Actual",yaxis_title="Predicted",
                             template="plotly_white",height=400)


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 3: MODEL TRAINING ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _build_clf_pipeline(model):
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("clf",model)])

def _build_reg_pipeline(model):
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("reg",model)])

def _prepare_X(df):
    feats=[c for c in FEATURE_COLS if c in df.columns]
    return df[feats].copy(), feats

# Path for cached eval metadata (so we skip re-eval on reload)
EVAL_CACHE_PATH = MODELS_DIR / "eval_cache.pkl"

def train_models(df):
    """Train classifier + regressor. Return (clf, reg, clf_eval, reg_eval, le, feats).
    Uses reduced estimator counts (100) for fast first-run training on small datasets.
    """
    X, feats = _prepare_X(df)
    le = LabelEncoder()
    y_cls = le.fit_transform(df["recovery_class"])
    y_rat = df["recovery_ratio"].values

    X_tr,X_te,yc_tr,yc_te = train_test_split(X,y_cls,test_size=0.2,random_state=42,stratify=y_cls)
    _,_,yr_tr,yr_te       = train_test_split(X,y_rat,test_size=0.2,random_state=42)

    # Reduced estimators: 100 trees is plenty for 400 training rows and trains ~2x faster
    models_clf = {
        "Logistic Regression": LogisticRegression(max_iter=500,class_weight="balanced",
                                                   random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100,max_depth=8,
                                                       min_samples_leaf=3,class_weight="balanced",random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=80,learning_rate=0.1,
                                                           max_depth=4,random_state=42),
    }
    best_f1=-1; best_name=None; best_clf=None; all_res={}
    for name,mdl in models_clf.items():
        p=_build_clf_pipeline(mdl); p.fit(X_tr,yc_tr); yp=p.predict(X_te)
        try: proba=p.predict_proba(X_te); auc=round(roc_auc_score(yc_te,proba,multi_class="ovr",average="macro"),4)
        except: auc="N/A"
        f1=f1_score(yc_te,yp,average="weighted",zero_division=0)
        all_res[name]={"model_name":name,"pipeline":p,
            "accuracy":round(accuracy_score(yc_te,yp),4),
            "precision":round(precision_score(yc_te,yp,average="weighted",zero_division=0),4),
            "recall":round(recall_score(yc_te,yp,average="weighted",zero_division=0),4),
            "f1_score":round(f1,4),"roc_auc":auc,
            "confusion_matrix":confusion_matrix(yc_te,yp),
            "classification_report":classification_report(yc_te,yp,target_names=le.classes_,output_dict=True),
            "y_test":yc_te,"y_pred":yp}
        if f1>best_f1: best_f1=f1; best_name=name; best_clf=p

    rf_model=all_res["Random Forest"]["pipeline"].named_steps["clf"]
    feat_imp=pd.Series(rf_model.feature_importances_,index=feats).sort_values(ascending=False)
    clf_eval=dict(all_res[best_name]); clf_eval["best_model_name"]=best_name
    clf_eval["feature_importances"]=feat_imp
    clf_eval["all_results"]={k:{kk:vv for kk,vv in v.items() if kk!="pipeline"} for k,v in all_res.items()}

    reg=_build_reg_pipeline(RandomForestRegressor(n_estimators=100,max_depth=8,min_samples_leaf=3,random_state=42))
    reg.fit(X_tr,yr_tr); yp_r=np.clip(reg.predict(X_te),0,1)
    reg_eval={"mae":round(mean_absolute_error(yr_te,yp_r),4),
              "mse":round(mean_squared_error(yr_te,yp_r),4),
              "rmse":round(np.sqrt(mean_squared_error(yr_te,yp_r)),4),
              "r2":round(r2_score(yr_te,yp_r),4),"y_test":yr_te,"y_pred":yp_r}

    # Persist models + eval so next load is instant
    joblib.dump(best_clf,CLASSIFIER_PATH); joblib.dump(reg,REGRESSOR_PATH)
    joblib.dump(le,LABEL_ENC_PATH); joblib.dump(feats,FEAT_NAMES_PATH)
    # Cache eval metadata separately (strip non-serialisable pipeline refs)
    eval_cache = {
        "clf_eval": {k:v for k,v in clf_eval.items() if k!="pipeline"},
        "reg_eval": reg_eval,
    }
    joblib.dump(eval_cache, EVAL_CACHE_PATH)
    return best_clf,reg,clf_eval,reg_eval,le,feats

def load_models_from_disk():
    try:
        return (joblib.load(CLASSIFIER_PATH),joblib.load(REGRESSOR_PATH),
                joblib.load(LABEL_ENC_PATH),joblib.load(FEAT_NAMES_PATH))
    except: return None,None,None,None

def models_exist():
    return all(p.exists() for p in [CLASSIFIER_PATH,REGRESSOR_PATH,LABEL_ENC_PATH,FEAT_NAMES_PATH])


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 4: PREDICTION ENGINE ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class RecoveryPredictor:
    def __init__(self): self.clf=self.reg=self.le=self.feats=None; self._ok=False

    def load(self,clf,reg,le,feats): self.clf=clf; self.reg=reg; self.le=le; self.feats=feats; self._ok=True

    def predict_single(self,rec:dict)->dict:
        row={f:rec.get(f,0) for f in self.feats}
        X=pd.DataFrame([row])
        enc=self.clf.predict(X)[0]; lbl=self.le.inverse_transform([enc])[0]
        try:
            proba=self.clf.predict_proba(X)[0]
            prob_dict={self.le.inverse_transform([i])[0]:round(float(p),4) for i,p in enumerate(proba)}
        except: prob_dict={lbl:1.0}
        ratio=float(np.clip(self.reg.predict(X)[0],0,1))
        return {"recovery_class":lbl,"recovery_probability":round(prob_dict.get(lbl,0)*100,1),
                "recovery_ratio":round(ratio,4),"predicted_recovery_pct":round(ratio*100,1),
                "probability_breakdown":prob_dict}

    def predict_batch(self,df:pd.DataFrame)->pd.DataFrame:
        avail=[c for c in self.feats if c in df.columns]
        X=df[avail].copy()
        for c in self.feats:
            if c not in X.columns: X[c]=0
        X=X[self.feats]
        encs=self.clf.predict(X); df["pred_recovery_class"]=self.le.inverse_transform(encs)
        try:
            proba=self.clf.predict_proba(X)
            hi=list(self.le.classes_).index("High Recovery") if "High Recovery" in self.le.classes_ else 0
            df["pred_high_recovery_prob"]=proba[:,hi]
        except: df["pred_high_recovery_prob"]=0.5
        ratios=np.clip(self.reg.predict(X),0,1)
        df["pred_recovery_ratio"]=ratios; df["pred_recovery_pct"]=(ratios*100).round(1)
        return df


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 5: RECOVERY AGENT ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

ACTION_TEMPLATES = {
    "High Recovery":{
        "HIGH":   "IMMEDIATE ACTION: Outstanding amount is significant. Assign a dedicated recovery officer. Issue a formal demand notice and schedule repayment negotiation within 7 days. Offer a structured settlement plan or EMI arrangement.",
        "MEDIUM": "STANDARD FOLLOW-UP: Good recovery potential. Send automated payment reminder with repayment link. Follow up by phone within 14 days if no response.",
        "LOW":    "ROUTINE MONITORING: High probability of recovery. Send a courteous reminder email. No escalation needed. Schedule 30-day review.",
    },
    "Medium Recovery":{
        "HIGH":   "URGENT ESCALATION: Medium recovery probability with high outstanding. Escalate to senior recovery team. Offer one-time settlement discount (up to 10%). Set 10-day deadline for response.",
        "MEDIUM": "ACTIVE RECOVERY: Moderate chance. Send formal reminder. Offer flexible repayment. Follow up by phone every 7 days. Consider small discount if borrower engages.",
        "LOW":    "SOFT FOLLOW-UP: Moderate potential on smaller account. Automated reminder sequence (3 emails over 21 days). No escalation unless no response within 30 days.",
    },
    "Low Recovery":{
        "HIGH":   "CRITICAL ACTION: Low recovery probability with large outstanding. Refer immediately to external collections agency. Evaluate legal recovery options. Offer final settlement discount (up to 20%) as last resort.",
        "MEDIUM": "COLLECTIONS REFERRAL: Low recovery probability. Refer to collections team. Issue legal notice. Offer time-limited settlement. Set 15-day deadline.",
        "LOW":    "WRITE-OFF EVALUATION: Low probability on small balance. Evaluate cost-benefit. Consider write-off if recovery cost exceeds potential return. Send final courtesy notice.",
    },
}

class RecoveryAgent:
    def __init__(self,predictor:RecoveryPredictor): self.p=predictor

    def priority_score(self,rc,prob,out,grade,delinq,dti)->float:
        r = 1.0-prob if rc=="Low Recovery" else (0.5+(1-prob)*0.3 if rc=="Medium Recovery" else (1-prob)*0.4)
        r=np.clip(r,0,1)
        return round(np.clip(
            0.35*r*100 + 0.30*min(out/40000,1)*100 +
            0.20*((grade-1)/6)*100 + 0.10*min(delinq/5,1)*100 +
            0.05*min(dti/50,1)*100, 0,100),1)

    def priority_level(self,s)->str:
        return "HIGH" if s>=65 else ("MEDIUM" if s>=35 else "LOW")

    def analyse(self,rec:dict)->dict:
        pred=self.p.predict_single(rec)
        loan=float(rec.get("loan_amnt",0))
        out=float(rec.get("outstanding_amount",max(0,loan-float(rec.get("total_rec_prncp",0)))))
        prob=pred["recovery_probability"]/100
        grade=int(rec.get("grade_encoded",4)); delinq=int(rec.get("delinq_2yrs",0)); dti=float(rec.get("dti",20))
        score=self.priority_score(pred["recovery_class"],prob,out,grade,delinq,dti)
        level=self.priority_level(score)
        action=ACTION_TEMPLATES.get(pred["recovery_class"],{}).get(level,"Monitor and review in 30 days.")
        acct_id=rec.get("id",rec.get("member_id","N/A"))
        grade_lbl={1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G"}.get(grade,"N/A")
        expl="\n".join([
            f"• Recovery Class: {pred['recovery_class']} (predicted ≈ {pred['predicted_recovery_pct']}%)",
            f"• Priority Score: {score}/100 → {level}",
            f"• Outstanding: ₹{out:,.0f}",
            f"• Grade: {grade_lbl} – {'Low' if grade<=2 else 'Moderate' if grade<=4 else 'High'} risk",
            f"• DTI: {dti:.1f}% – {'OK' if dti<=30 else 'Elevated' if dti<=40 else 'High'}",
            f"• Delinquencies (2yr): {delinq}",
        ])
        prob_bd=pred["probability_breakdown"]
        for cls,pv in sorted(prob_bd.items(),key=lambda x:-x[1]):
            expl+=f"\n  - {cls}: {pv*100:.1f}%"
        pem={"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(level,"⚪")
        report="\n".join([
            "="*60,"  REVENUE RECOVERY AGENT — ACCOUNT ANALYSIS REPORT","="*60,
            f"  Account ID       : {str(acct_id)[-8:]}","  (masked for privacy)",
            f"  Loan Amount      : ₹{loan:,.0f}",
            f"  Outstanding Amt  : ₹{out:,.0f}","-"*60,
            f"  Recovery Class   : {pred['recovery_class']}",
            f"  Recovery Prob.   : {pred['recovery_probability']:.1f}%",
            f"  Predicted Recov. : {pred['predicted_recovery_pct']:.1f}% of loan","-"*60,
            f"  Priority Score   : {score:.1f}/100",
            f"  Priority Level   : {pem} {level}","-"*60,
            "  RECOMMENDED ACTION:","",
        ]+[f"  {l}" for l in textwrap.wrap(action,56)]+[
            "","  ANALYSIS DETAILS:","",
        ]+[f"  {l}" for l in expl.split("\n")]+[
            "","="*60,
            "  ⚠ System-generated recommendation. Requires officer review.",
            "="*60,
        ])
        return {"account_id":acct_id,"loan_amount":loan,"outstanding_amount":out,
                "recovery_class":pred["recovery_class"],"recovery_probability":pred["recovery_probability"],
                "predicted_recovery_pct":pred["predicted_recovery_pct"],
                "probability_breakdown":prob_bd,"priority_score":score,
                "priority_level":level,"recommended_action":action,
                "explanation":expl,"report":report}

    def batch_analyse(self,df:pd.DataFrame)->pd.DataFrame:
        df2=self.p.predict_batch(df.copy())
        if "outstanding_amount" not in df2.columns:
            df2["outstanding_amount"]=(df2["loan_amnt"]-df2["total_rec_prncp"]).clip(lower=0)
        scores,levels,actions=[],[],[]
        for i in range(len(df2)):
            rc=df2["pred_recovery_class"].iloc[i]
            pb=float(df2["pred_high_recovery_prob"].iloc[i])
            out=float(df2["outstanding_amount"].iloc[i])
            g=int(df2.get("grade_encoded",pd.Series([4]*len(df2))).iloc[i])
            d=int(df2.get("delinq_2yrs",pd.Series([0]*len(df2))).iloc[i])
            dt=float(df2.get("dti",pd.Series([20.0]*len(df2))).iloc[i])
            s=self.priority_score(rc,pb,out,g,d,dt)
            lv=self.priority_level(s)
            ac=ACTION_TEMPLATES.get(rc,{}).get(lv,"Monitor account.")
            scores.append(s); levels.append(lv); actions.append(ac)
        df2["priority_score"]=scores; df2["priority_level"]=levels; df2["recommended_action"]=actions
        return df2


# ═══════════════════════════════════════════════════════════════════════════
# ── SECTION 6: STREAMLIT DASHBOARD ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── Page config MUST be first Streamlit call ─────────────────────────────────
st.set_page_config(page_title="Revenue Recovery Agent",page_icon="💰",
                   layout="wide",initial_sidebar_state="expanded")

STYLE = """
<style>
.main{background-color:#f8fafc}
.metric-card{background:white;border-radius:10px;padding:18px 20px;
  border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,.06);text-align:center}
.metric-card .mv{font-size:1.5rem;font-weight:700;color:#1e3a5f;margin:4px 0}
.metric-card .ml{font-size:.80rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.sh{font-size:1.2rem;font-weight:600;color:#1e3a5f;border-left:4px solid #3b82f6;
    padding-left:10px;margin:18px 0 10px 0}
.rb{background:#0f172a;color:#e2e8f0;font-family:monospace;font-size:.80rem;
    border-radius:8px;padding:16px;white-space:pre-wrap;line-height:1.55}
footer{visibility:hidden}
section[data-testid="stSidebar"]{background:#1e3a5f}
section[data-testid="stSidebar"] *{color:white!important}
</style>"""

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_cur(v,s="₹"):
    if pd.isna(v): return "N/A"
    return f"{s}{v/1e6:.2f}M" if abs(v)>=1e6 else f"{s}{v:,.0f}"

def fmt_pct(v,d=1): return "N/A" if pd.isna(v) else f"{v:.{d}f}%"
def fmt_num(v): return "N/A" if pd.isna(v) else f"{int(v):,}"

def mcard(label,val,delta=None,col="#1e3a5f"):
    dh=f'<div style="font-size:.73rem;color:#64748b;">{delta}</div>' if delta else ""
    st.markdown(f'<div class="metric-card"><div class="ml">{label}</div>'
                f'<div class="mv" style="color:{col};">{val}</div>{dh}</div>',
                unsafe_allow_html=True)

def mask_id(s): s=str(s); return ("*"*(len(s)-4)+s[-4:]) if len(s)>4 else s

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_df():
    return load_and_preprocess()

@st.cache_resource(show_spinner=False)
def _cached_models():
    """Instant load from disk if saved; otherwise train once then save."""
    if models_exist() and EVAL_CACHE_PATH.exists():
        clf,reg,le,feats = load_models_from_disk()
        if clf is not None:
            ec = joblib.load(EVAL_CACHE_PATH)
            return clf, reg, ec["clf_eval"], ec["reg_eval"], le, feats
    return train_models(_cached_df())


# ════════════════════════════════════════════════════════════════════════════
# MAIN — wraps everything so Streamlit renders sidebar before any heavy work
# ════════════════════════════════════════════════════════════════════════════
def _page_dashboard(df, kpis):
    st.title("📊 Revenue Recovery Dashboard")
    st.caption("Overview of recovery performance across all charged-off loan accounts.")
    c1,c2,c3,c4=st.columns(4)
    with c1: mcard("Total Accounts",fmt_num(kpis["Total Accounts"]))
    with c2: mcard("Total Loan Amount",fmt_cur(kpis["Total Loan Amount (₹)"]))
    with c3: mcard("Total Recovered",fmt_cur(kpis["Total Amount Recovered (₹)"]),col="#16a34a")
    with c4: mcard("Outstanding Amount",fmt_cur(kpis["Total Outstanding Amount (₹)"]),col="#dc2626")
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: mcard("Recovery Rate",fmt_pct(kpis["Recovery Rate (%)"]),col="#3b82f6")
    with c2: mcard("High Risk Accounts",fmt_num(kpis["High Risk Accounts"]),"Low Recovery",col="#dc2626")
    with c3: mcard("Avg Loan Amount",fmt_cur(kpis["Avg Loan Amount (₹)"]))
    with c4: mcard("Avg Interest Rate",fmt_pct(kpis["Avg Interest Rate (%)"]))
    st.divider()
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_gauge(kpis["Recovery Rate (%)"]),use_container_width=True)
    with c2: st.plotly_chart(plot_recovery_class_dist(df),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_loan_dist(df),use_container_width=True)
    with c2: st.plotly_chart(plot_recovery_ratio_dist(df),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_loan_vs_recovered(df),use_container_width=True)
    with c2: st.plotly_chart(plot_top_risk(df),use_container_width=True)

def _page_account_analysis(df):
    st.title("🔍 Account Analysis")
    with st.expander("🔧 Filter Accounts",expanded=True):
        c1,c2,c3=st.columns(3)
        with c1: sg=st.multiselect("Grade",sorted(df["grade"].dropna().unique()),default=sorted(df["grade"].dropna().unique()))
        with c2: sc=st.multiselect("Recovery Class",sorted(df["recovery_class"].dropna().unique()),default=sorted(df["recovery_class"].dropna().unique()))
        with c3: sp=st.multiselect("Purpose",sorted(df["purpose"].dropna().unique()),default=sorted(df["purpose"].dropna().unique())[:5])
        c1,c2=st.columns(2)
        with c1: lr=st.slider("Loan Amount (₹)",int(df["loan_amnt"].min()),int(df["loan_amnt"].max()),(int(df["loan_amnt"].min()),int(df["loan_amnt"].max())))
        with c2: rr=st.slider("Interest Rate (%)",float(df["int_rate"].min()),float(df["int_rate"].max()),(float(df["int_rate"].min()),float(df["int_rate"].max())))
    msk=(df["grade"].isin(sg)&df["recovery_class"].isin(sc)&df["purpose"].isin(sp)&
         df["loan_amnt"].between(*lr)&df["int_rate"].between(*rr))
    dff=df[msk].copy()
    st.markdown(f"**{len(dff):,} of {len(df):,} accounts**")
    c1,c2,c3,c4=st.columns(4)
    with c1: mcard("Filtered Accounts",fmt_num(len(dff)))
    with c2: mcard("Total Loan",fmt_cur(dff["loan_amnt"].sum()))
    with c3: mcard("Total Recovered",fmt_cur(dff["total_pymnt"].sum()),col="#16a34a")
    with c4: mcard("Avg Recovery Rate",fmt_pct(dff["recovery_ratio"].mean()*100 if len(dff)>0 else 0),col="#3b82f6")
    st.markdown("<br>",unsafe_allow_html=True)
    if len(dff)>0:
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(plot_recovery_grade(dff),use_container_width=True)
        with c2: st.plotly_chart(plot_outstanding_grade(dff),use_container_width=True)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(plot_int_rate_box(dff),use_container_width=True)
        with c2: st.plotly_chart(plot_dti_box(dff),use_container_width=True)
    dcols=["id","grade","loan_amnt","int_rate","term","purpose","annual_inc","dti",
           "total_pymnt","outstanding_amount","recovery_ratio","recovery_class"]
    acols=[c for c in dcols if c in dff.columns]
    dv=dff[acols].copy()
    if "id" in dv.columns: dv["id"]=dv["id"].astype(str).apply(mask_id)
    if "recovery_ratio" in dv.columns: dv["recovery_ratio"]=dv["recovery_ratio"].map(lambda x:f"{x:.1%}")
    st.markdown('<div class="sh">Account Records</div>',unsafe_allow_html=True)
    st.dataframe(dv.head(200),use_container_width=True,hide_index=True)


def _render_prediction(rec, agent):
    res=agent.analyse(rec)
    pc=PRIORITY_COLOURS.get(res["priority_level"],"#64748b")
    rc=RECOVERY_COLOURS.get(res["recovery_class"],"#64748b")
    em=PRIORITY_EMOJI.get(res["priority_level"],"⚪")
    st.divider()
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: mcard("Loan Amount",fmt_cur(res["loan_amount"]))
    with c2: mcard("Outstanding",fmt_cur(res["outstanding_amount"]),col="#dc2626")
    with c3: mcard("Recovery Class",res["recovery_class"].replace(" Recovery",""),col=rc)
    with c4: mcard("Recovery Prob",f"{res['recovery_probability']:.1f}%",col="#3b82f6")
    with c5: mcard("Priority",f"{em} {res['priority_level']}",f"Score: {res['priority_score']:.0f}/100",col=pc)
    st.markdown("<br>",unsafe_allow_html=True)
    pb=res["probability_breakdown"]
    if pb:
        c1,c2=st.columns(2)
        with c1:
            cm2={"High Recovery":"#22c55e","Medium Recovery":"#f59e0b","Low Recovery":"#ef4444"}
            fig=go.Figure(go.Bar(x=list(pb.keys()),y=[v*100 for v in pb.values()],
                marker_color=[cm2.get(k,"#3b82f6") for k in pb.keys()],
                text=[f"{v*100:.1f}%" for v in pb.values()],textposition="outside"))
            fig.update_layout(title="Recovery Class Probabilities",yaxis_title="Probability (%)",
                              template="plotly_white",height=320,yaxis=dict(range=[0,115]))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2=go.Figure(go.Indicator(mode="gauge+number",value=res["priority_score"],
                title={"text":"Priority Score / 100"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":pc},
                       "steps":[{"range":[0,35],"color":"#d1fae5"},
                                {"range":[35,65],"color":"#fef3c7"},
                                {"range":[65,100],"color":"#fecaca"}]}))
            fig2.update_layout(height=320)
            st.plotly_chart(fig2,use_container_width=True)
    st.markdown('<div class="sh">💡 Recommended Action</div>',unsafe_allow_html=True)
    st.info(res["recommended_action"])
    st.markdown('<div class="sh">📄 Full Agent Report</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="rb">{res["report"]}</div>',unsafe_allow_html=True)
    with st.expander("🔬 Analysis Details"):
        st.text(res["explanation"])


def _page_prediction(df, agent):
    st.title("🤖 Recovery Prediction & Agent Report")
    tab1,tab2=st.tabs(["📁 Select Existing Account","✏️ Enter Custom Account"])
    with tab1:
        id_col="id" if "id" in df.columns else df.columns[0]
        ids=df[id_col].dropna().astype(str).unique().tolist()
        ca,cb=st.columns([2,1])
        with ca: sel=st.selectbox("Select Account ID",ids)
        with cb:
            st.markdown("<br>",unsafe_allow_html=True)
            go_btn=st.button("▶ Analyse Account",type="primary",key="btn_exist")
        if go_btn:
            row=df[df[id_col].astype(str)==str(sel)]
            if len(row)>0: _render_prediction(row.iloc[0].to_dict(), agent)
            else: st.warning("Account not found.")
    with tab2:
        st.markdown("**Enter account details:**")
        c1,c2,c3=st.columns(3)
        with c1:
            la=st.number_input("Loan Amount (₹)",1000,40000,15000,500)
            ir=st.number_input("Interest Rate (%)",5.0,35.0,14.0,.5)
            ai=st.number_input("Annual Income (₹)",10000,500000,65000,5000)
        with c2:
            ts=st.selectbox("Loan Term",["36 months","60 months"])
            gs=st.selectbox("Loan Grade",["A","B","C","D","E","F","G"],index=2)
            dt=st.number_input("DTI (%)",0.0,60.0,22.0,.5)
        with c3:
            hs=st.selectbox("Home Ownership",["OWN","RENT","MORTGAGE","OTHER"],index=1)
            es=st.selectbox("Employment Length",list(EMP_MAP.keys()),index=5)
            dq=st.number_input("Delinquencies (2yr)",0,10,0)
        c1,c2,c3=st.columns(3)
        with c1:
            tp=st.number_input("Total Payment Received (₹)",0,40000,2000,100)
            trp=st.number_input("Principal Recovered (₹)",0,40000,1500,100)
        with c2:
            ru=st.number_input("Revolving Util (%)",0.0,120.0,55.0,1.0)
            inq=st.number_input("Inquiries (6m)",0,10,1)
        with c3:
            pr=st.number_input("Public Records",0,10,0)
            oa=st.number_input("Open Accounts",1,50,12)
        vs=st.selectbox("Verification Status",["Not Verified","Source Verified","Verified"],index=1)
        pu=st.selectbox("Loan Purpose",sorted(df["purpose"].dropna().unique()))
        if st.button("▶ Run Prediction",type="primary",key="btn_custom"):
            tm=36 if "36" in ts else 60
            inst=(la*(ir/100/12)/(1-(1+ir/100/12)**(-tm)))
            pts=sorted(df["purpose"].dropna().unique())
            pe=pts.index(pu) if pu in pts else 0
            cdict={
                "id":"CUSTOM","loan_amnt":float(la),"funded_amnt":float(la),"int_rate":float(ir),
                "installment":round(inst,2),"annual_inc":float(ai),"dti":float(dt),
                "delinq_2yrs":int(dq),"inq_last_6mths":int(inq),"open_acc":int(oa),"pub_rec":int(pr),
                "revol_bal":float(ru*1000),"revol_util":float(ru),"total_acc":int(oa+5),
                "total_pymnt":float(tp),"total_rec_prncp":float(trp),"total_rec_int":float(tp-trp),
                "total_rec_late_fee":0.0,"last_pymnt_amnt":round(inst,2),
                "collections_12_mths_ex_med":0,"acc_now_delinq":0,"tot_coll_amt":0.0,
                "tot_cur_bal":float(ai*.5),"total_rev_hi_lim":float(ai*.3),
                "term_months":float(tm),"emp_length_years":float(EMP_MAP.get(es,5)),
                "grade_encoded":float(GRADE_MAP.get(gs,4)),
                "home_ownership_encoded":float(HOME_MAP.get(hs,3)),
                "verification_status_encoded":float(VERIF_MAP.get(vs,1)),
                "purpose_encoded":float(pe),
                "months_outstanding":float(max(0,la-trp)/max(inst,1)),
                "outstanding_amount":float(max(0,la-trp)),
                "high_dti_flag":int(dt>35),"high_int_rate_flag":int(ir>20),
                "no_delinq_flag":int(dq==0),
                "payment_completeness":min(tp/max(inst*tm,1),1.0),
                "income_to_loan_ratio":min(ai/max(la,1),50),
                "revol_pressure":min(ru/100,2.0),
            }
            _render_prediction(cdict, agent)


def _page_analytics(df):
    st.title("📈 Analytics")
    st.markdown('<div class="sh">Recovery by Loan Characteristics</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_recovery_grade(df),use_container_width=True)
    with c2: st.plotly_chart(plot_term_recovery(df),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_outstanding_grade(df),use_container_width=True)
    with c2: st.plotly_chart(plot_home_ownership(df),use_container_width=True)
    st.markdown('<div class="sh">Borrower Profile</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_income_scatter(df),use_container_width=True)
    with c2: st.plotly_chart(plot_verif_recovery(df),use_container_width=True)
    st.markdown('<div class="sh">Recovery by Loan Purpose</div>',unsafe_allow_html=True)
    st.plotly_chart(plot_recovery_purpose(df),use_container_width=True)
    st.markdown('<div class="sh">Risk Factors</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(plot_int_rate_box(df),use_container_width=True)
    with c2: st.plotly_chart(plot_dti_box(df),use_container_width=True)
    st.markdown('<div class="sh">Correlation Analysis</div>',unsafe_allow_html=True)
    st.plotly_chart(plot_corr_heatmap(df),use_container_width=True)
    st.markdown('<div class="sh">Grade × Purpose Recovery Heatmap</div>',unsafe_allow_html=True)
    piv=df.pivot_table(values="recovery_ratio",index="grade",columns="purpose",aggfunc="mean")
    st.plotly_chart(px.imshow(piv,text_auto=".2f",color_continuous_scale="RdYlGn",
        title="Avg Recovery Ratio: Grade × Purpose",aspect="auto"
        ).update_layout(template="plotly_white",height=400),use_container_width=True)


def _page_model_performance(clf_eval, reg_eval, le):
    st.title("🧠 Model Performance")
    st.markdown('<div class="sh">Classifier — Recovery Class Prediction</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: mcard("Best Model",clf_eval.get("best_model_name","Random Forest"),col="#3b82f6")
    with c2: mcard("Accuracy",fmt_pct(clf_eval["accuracy"]*100),col="#16a34a")
    with c3: mcard("F1 Score",fmt_pct(clf_eval["f1_score"]*100),col="#3b82f6")
    with c4:
        auc=clf_eval.get("roc_auc","N/A")
        mcard("ROC-AUC",str(auc) if auc=="N/A" else fmt_pct(float(auc)*100),col="#8b5cf6")
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: mcard("Precision",fmt_pct(clf_eval["precision"]*100))
    with c2: mcard("Recall",fmt_pct(clf_eval["recall"]*100))
    st.markdown("<br>",unsafe_allow_html=True)
    st.plotly_chart(plot_confusion(clf_eval["confusion_matrix"],list(le.classes_)),use_container_width=True)
    rpt=clf_eval.get("classification_report",{})
    rows=[{"Class":c,"Precision":f"{rpt[c]['precision']:.3f}","Recall":f"{rpt[c]['recall']:.3f}",
           "F1":f"{rpt[c]['f1-score']:.3f}","Support":int(rpt[c]["support"])}
          for c in le.classes_ if c in rpt]
    if rows:
        st.markdown('<div class="sh">Per-Class Report</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    ar=clf_eval.get("all_results",{})
    if ar:
        st.markdown('<div class="sh">Model Comparison</div>',unsafe_allow_html=True)
        comp=[{"Model":n,"Accuracy":f"{r['accuracy']:.4f}","Precision":f"{r['precision']:.4f}",
               "Recall":f"{r['recall']:.4f}","F1":f"{r['f1_score']:.4f}",
               "ROC-AUC":str(r['roc_auc'])} for n,r in ar.items()]
        st.dataframe(pd.DataFrame(comp),use_container_width=True,hide_index=True)
    fi=clf_eval.get("feature_importances")
    if fi is not None:
        st.markdown('<div class="sh">Feature Importance</div>',unsafe_allow_html=True)
        st.plotly_chart(plot_feat_imp(fi),use_container_width=True)
    st.divider()
    st.markdown('<div class="sh">Regressor — Recovery Ratio Prediction</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: mcard("MAE",f"{reg_eval['mae']:.4f}")
    with c2: mcard("RMSE",f"{reg_eval['rmse']:.4f}")
    with c3: mcard("R²",f"{reg_eval['r2']:.4f}",col="#16a34a" if reg_eval['r2']>0.7 else "#d97706")
    with c4: mcard("MSE",f"{reg_eval['mse']:.4f}")
    st.markdown("<br>",unsafe_allow_html=True)
    st.plotly_chart(plot_pred_scatter(reg_eval["y_pred"],reg_eval["y_test"]),use_container_width=True)


def _page_data_explorer(df):
    st.title("📋 Data Explorer")
    c1,c2,c3,c4=st.columns(4)
    with c1: mcard("Total Records",fmt_num(len(df)))
    with c2: mcard("Total Columns",fmt_num(len(df.columns)))
    with c3: mcard("Numeric Cols",fmt_num(df.select_dtypes(include=np.number).shape[1]))
    with c4: mcard("Categorical Cols",fmt_num(df.select_dtypes(include="object").shape[1]))
    st.markdown("<br>",unsafe_allow_html=True)
    dcols=["id","grade","loan_amnt","int_rate","term","purpose","annual_inc","dti",
           "total_pymnt","outstanding_amount","recovery_ratio","recovery_class"]
    sel_cols=st.multiselect("Select Columns",df.columns.tolist(),
                            default=[c for c in dcols if c in df.columns])
    if sel_cols:
        dv=df[sel_cols].copy()
        if "id" in dv.columns: dv["id"]=dv["id"].astype(str).apply(mask_id)
        if "recovery_ratio" in dv.columns: dv["recovery_ratio"]=dv["recovery_ratio"].map(lambda x:f"{x:.1%}")
        srch=st.text_input("🔎 Search","")
        if srch:
            msk2=dv.apply(lambda col:col.astype(str).str.contains(srch,case=False,na=False)).any(axis=1)
            dv=dv[msk2]; st.markdown(f"Found **{len(dv)}** records")
        st.dataframe(dv,use_container_width=True,height=450)
    st.markdown('<div class="sh">Descriptive Statistics</div>',unsafe_allow_html=True)
    nc=["loan_amnt","int_rate","annual_inc","dti","total_pymnt","outstanding_amount","recovery_ratio","revol_util","delinq_2yrs"]
    st.dataframe(df[[c for c in nc if c in df.columns]].describe().round(3),use_container_width=True)
    st.markdown('<div class="sh">Missing Values</div>',unsafe_allow_html=True)
    miss=df.isnull().sum(); miss=miss[miss>0]
    if len(miss)>0:
        st.dataframe(pd.DataFrame({"Column":miss.index,"Missing":miss.values,"Pct":(miss.values/len(df)*100).round(2)}),
                     use_container_width=True,hide_index=True)
    else:
        st.success("✅ No missing values after preprocessing.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — sidebar renders first, then data/models load lazily
# ════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(STYLE, unsafe_allow_html=True)

    # Sidebar renders instantly (no data needed)
    with st.sidebar:
        st.markdown("## 💰 Revenue Recovery Agent")
        st.markdown("---")
        page=st.radio("Navigate",[
            "📊 Dashboard","🔍 Account Analysis",
            "🤖 Recovery Prediction","📈 Analytics",
            "🧠 Model Performance","📋 Data Explorer"],
            label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Dataset:** LendingClub Charged-Off Loans")
        st.markdown("**Project:** IBM SkillsBuild Internship")
        st.markdown("---")
        st.markdown("*Revenue Recovery Agent v1.0*")

    # Load data — fast (CSV parse + feature engineering, ~1s)
    with st.spinner("⏳ Loading dataset…"):
        df = _cached_df()

    # Load/train models — fast on reload (disk cache), ~10s on first run
    with st.spinner("⚙️ Loading ML models… (first run trains once, ~10s)"):
        clf, reg, clf_eval, reg_eval, le, feats = _cached_models()

    # Build predictor + agent (in-memory only, instant)
    predictor = RecoveryPredictor()
    predictor.load(clf, reg, le, feats)
    agent = RecoveryAgent(predictor)
    kpis = compute_kpis(df)

    # Route to selected page
    if   page=="📊 Dashboard":          _page_dashboard(df, kpis)
    elif page=="🔍 Account Analysis":   _page_account_analysis(df)
    elif page=="🤖 Recovery Prediction":_page_prediction(df, agent)
    elif page=="📈 Analytics":          _page_analytics(df)
    elif page=="🧠 Model Performance":  _page_model_performance(clf_eval, reg_eval, le)
    elif page=="📋 Data Explorer":      _page_data_explorer(df)

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#94a3b8;font-size:.76rem;'>"
                "Revenue Recovery Agent · IBM SkillsBuild BharatCares · "
                "Python | Scikit-learn | Streamlit | Plotly</div>",
                unsafe_allow_html=True)


main()
