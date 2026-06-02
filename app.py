import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
import lime
import plotly.express as px
import lime.lime_tabular
import joblib
import os
import time
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, precision_recall_curve, auc,
                             classification_report)
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# ==================== LOGIN (CLASSIC CENTERED CARD) ====================
def get_logo_path():
    possible_paths = ["BrandLogoImage.png"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def check_login():
    # ---------- USER DATABASE (add as many as you want) ----------
    # In production, hash passwords and store them securely.
    USERS = {
        "admin": "admin123",
        "analyst": "ana123",
        "auditor": "audit2025",
        "manager": "risk123",
        "john_doe": "securepass",
        "finance_team": "fin2026",
    }
    if st.session_state.get("logged_in", False):
        return True

    # Clean classic background
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        }
        /* Center the card vertically and horizontally */
        .block-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 0 !important;

        }

        /* Logo styling */
        .login-logo img {
            max-width: 200px;
        }
        /* Title */
        .login-title {
            font-size: 28px;
            font-weight: 700;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 4px;
        }
        .login-sub {
            font-size: 14px;
            color: #5a6e7c;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        /* Divider */
        .login-divider {
            margin: 1rem 0;
            border-top: 1px solid #e9ecef;
        }
        /* Inputs */
        .stTextInput input {
            border-radius: 12px;
            border: 1px solid #ced4da;
            padding: 10px 14px;
            background: white;
        }
        .stTextInput input:focus {
            border-color: #1f4e79;
            box-shadow: 0 0 0 2px rgba(31,78,121,0.2);
        }
        /* Button */
        .stButton button {
            background-color: #1f4e79;
            color: white;
            border-radius: 40px;
            padding: 10px 0;
            font-weight: 600;
            border: none;
            transition: 0.2s;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #0e3a5c;
            transform: translateY(-2px);
        }
        /* Footer */
        .login-footer {
            font-size: 11px;
            color: #8c9aa8;
            text-align: center;
            margin-top: 1.5rem;

        }
    </style>
    """, unsafe_allow_html=True)

    # Center the form
    # Center the form using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # ----- Logo, centered, with adjustable size -----
        logo_path = get_logo_path()
        if logo_path and os.path.exists(logo_path):
            st.image(logo_path, width=300)  # ← change size here (pixels)
        else:
            st.markdown('<div class="login-title">FinGuard AI</div>', unsafe_allow_html=True)

        st.markdown('<div class="login-sub">Risk Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-divider"></div>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please fill all fields")
            else:
                # Check against user dictionary (no st.secrets)
                if username in USERS and USERS[username] == password:
                    with st.spinner("Authenticating..."):
                        time.sleep(0.6)
                    st.session_state["logged_in"] = True
                    st.success(f"Welcome, {username}! Redirecting...")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown('<p class="login-footer">Authorized users only · FinGuard AI</p>', unsafe_allow_html=True)

    return False


# Login Gate
if not check_login():
    st.stop()

# ------------------- Page configuration (once, at the top) -------------------
st.set_page_config(page_title="FinGaurd AI", layout="wide", page_icon="📈")

# ==================== CUSTOM CSS FOR WEB APP ====================
st.markdown("""
<style>
    /* Global styling */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    /* Title and headers */
    .css-10trblm.e16nr0p30 {
        color: #1f4e79;
        font-weight: 600;
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1f4e79;
        font-weight: 600;
    }
    /* Metric cards */
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: 0.2s;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Buttons */
    .stButton button {
        background-color: #1f4e79;
        color: white;
        border-radius: 8px;
        font-weight: 500;
        border: none;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #0e3a5c;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    /* Dataframe tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f3f4;
        border-radius: 16px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px;
        padding: 8px 16px;
        font-weight: 500;
        transition: 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e6e9ef;
    }
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #1f4e79;
    }
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f4e79;
    }
    /* Plotly charts */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR (only visible after login) ====================
logo_path = get_logo_path()
if logo_path:
    st.sidebar.image(logo_path, use_container_width=False, width=200)
    st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.info("👆 Please upload your dataset to continue.")
    st.stop()

# ------------------- Main App Title (without logo) -------------------
st.title("📈FinGuard AI – Fraud Risk Intelligence Platform")
st.markdown("---")

# ------------------- Session State -------------------
if 'logout' not in st.session_state:
    st.session_state.logout = False


# ------------------- Data Preprocessing -------------------
@st.cache_data
def load_and_preprocess(_file):
    df = pd.read_csv(_file)
    cols_to_drop = ['company_id', 'company_name']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    X = df.drop(columns=['anomaly_label'])
    y = df['anomaly_label']
    feature_names = X.columns.tolist()
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_res = pd.DataFrame(X_train_res, columns=X.columns)
    return df, X_train_res, y_train_res, X_test, y_test, feature_names, scaler


df, X_train_res, y_train_res, X_test, y_test, feature_names, scaler = load_and_preprocess(uploaded_file)

# ------------------- Models -------------------
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_paths = {
    'rf': os.path.join(model_dir, "rf.joblib"),
    'gb': os.path.join(model_dir, "gb.joblib"),
    'stack': os.path.join(model_dir, "stack.joblib")
}
force_retrain = st.sidebar.button("🔄 Force Retrain Models")


@st.cache_resource
def get_models():
    if all(os.path.exists(p) for p in model_paths.values()) and not force_retrain:
        return joblib.load(model_paths['rf']), joblib.load(model_paths['gb']), joblib.load(model_paths['stack'])
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    joblib.dump(rf, model_paths['rf'])
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gb.fit(X_train_res, y_train_res)
    joblib.dump(gb, model_paths['gb'])
    stack = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True, n_jobs=-1
    )
    stack.fit(X_train_res, y_train_res)
    joblib.dump(stack, model_paths['stack'])
    return rf, gb, stack


rf, gb, stack = get_models()


# ------------------- Predictions -------------------
@st.cache_data
def get_predictions():
    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)
    y_pred_stack = stack.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    y_proba_stack = stack.predict_proba(X_test)[:, 1]
    return y_pred_rf, y_pred_gb, y_pred_stack, y_proba_rf, y_proba_gb, y_proba_stack


y_pred_rf, y_pred_gb, y_pred_stack, y_proba_rf, y_proba_gb, y_proba_stack = get_predictions()


# ------------------- SHAP Precomputation -------------------
@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(rf)


@st.cache_data
def get_shap_values():
    explainer = get_shap_explainer()
    sample_size = min(500, len(X_test))
    sample = X_test.sample(sample_size, random_state=42)
    shap_values_all = explainer.shap_values(sample)
    if isinstance(shap_values_all, list):
        shap_values_for_class1 = shap_values_all[1]
    elif len(shap_values_all.shape) == 3:
        shap_values_for_class1 = shap_values_all[:, :, 1]
    else:
        shap_values_for_class1 = shap_values_all
    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]
    return sample, shap_values_for_class1, expected


shap_sample, shap_values, expected_val = get_shap_values()


# ------------------- LIME Explainer -------------------
@st.cache_resource
def get_lime_explainer():
    return lime.lime_tabular.LimeTabularExplainer(
        np.array(X_train_res),
        feature_names=feature_names,
        class_names=['Normal', 'Anomaly'],
        mode='classification'
    )


lime_explainer = get_lime_explainer()


# ------------------- PDF Helper with Logo Header -------------------
def add_logo_header(doc, styles, title):
    now = datetime.now()
    ts_str = f"Generated on: {now.strftime('%Y-%m-%d')} at {now.strftime('%H:%M:%S')}"
    full_title = f"{title}<br/><font size=8>{ts_str}</font>"
    logo_path = get_logo_path()
    if logo_path:
        try:
            logo_img = Image(logo_path, width=35 * mm, height=18 * mm)
            logo_container = Table([[logo_img]], colWidths=[40 * mm])
            logo_container.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
        except:
            logo_container = Paragraph("", styles['Normal'])
    else:
        logo_container = Paragraph("", styles['Normal'])
    title_para = Paragraph(f"<font size=16 color='#FFFFFF'><b>{full_title}</b></font>", styles['Normal'])
    header_table = Table([[logo_container, title_para]], colWidths=[50 * mm, doc.width - 50 * mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1f4e79")),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    return header_table


def add_heading(text, level=2):
    size = 12 if level == 2 else 14
    return Paragraph(f"<font size={size}><b>{text}</b></font>", getSampleStyleSheet()['Heading2'])


def add_figure(fig, width=450, height=220):
    img_data = BytesIO()
    fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    img_data.seek(0)
    return Image(img_data, width=width, height=height)


# ------------------- Main Analysis PDF -------------------
def generate_full_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=25)
    styles = getSampleStyleSheet()
    elements = []

    # Page 1
    elements.append(add_logo_header(doc, styles, "CONFIDENTIAL FRAUD ANALYSIS REPORT"))
    elements.append(Spacer(1, 8))
    elements.append(add_heading("Dataset Summary"))
    summary_data = [
        ["Metric", "Value"],
        ["Total Records", f"{df.shape[0]:,}"],
        ["Features", df.shape[1] - 1],
        ["Fraud Cases", f"{int(df['anomaly_label'].sum()):,}"],
        ["Normal Cases", f"{int((df['anomaly_label'] == 0).sum()):,}"],
        ["Fraud Ratio", f"{df['anomaly_label'].mean():.2%}"]
    ]
    summary_table = Table(summary_data, colWidths=[150, 250])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 10))

    elements.append(add_heading("Top Risk Indicators"))

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]

    top_data = [["Feature", "Importance"]]
    for i in indices:
        top_data.append([feature_names[i], f"{importances[i]:.3f}"])

    top_table = Table(top_data, colWidths=[200, 200])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(top_table)
    elements.append(Spacer(1, 10))

    elements.append(add_heading("Fraud Distribution"))
    fig1 = plt.figure(figsize=(5, 3))
    counts = df['anomaly_label'].value_counts()
    bars = plt.bar(['Normal', 'Fraud'], counts, color=['#2E86AB', '#A23B72'])
    plt.title('Class Distribution', fontsize=11, fontweight='bold')
    plt.ylabel('Count')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    elements.append(add_figure(fig1, width=400, height=180))
    elements.append(PageBreak())

    # Page 2
    elements.append(add_logo_header(doc, styles, "MODEL PERFORMANCE COMPARISON"))
    elements.append(Spacer(1, 8))
    elements.append(add_heading("Accuracy Metrics"))
    comp_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1"],
        ["Random Forest", f"{accuracy_score(y_test, y_pred_rf):.2%}", f"{precision_score(y_test, y_pred_rf):.2%}",
         f"{recall_score(y_test, y_pred_rf):.2%}", f"{f1_score(y_test, y_pred_rf):.2%}"],
        ["Gradient Boosting", f"{accuracy_score(y_test, y_pred_gb):.2%}", f"{precision_score(y_test, y_pred_gb):.2%}",
         f"{recall_score(y_test, y_pred_gb):.2%}", f"{f1_score(y_test, y_pred_gb):.2%}"],
        ["Stacking Ensemble", f"{accuracy_score(y_test, y_pred_stack):.2%}",
         f"{precision_score(y_test, y_pred_stack):.2%}", f"{recall_score(y_test, y_pred_stack):.2%}",
         f"{f1_score(y_test, y_pred_stack):.2%}"]
    ]
    comp_table = Table(comp_data, colWidths=[80, 75, 75, 75, 75])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(comp_table)
    elements.append(Spacer(1, 12))

    elements.append(add_heading("Accuracy Comparison"))
    fig2 = plt.figure(figsize=(5, 3))
    models = ['RF', 'GB', 'STACK']
    acc = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_gb), accuracy_score(y_test, y_pred_stack)]
    bars = plt.bar(models, acc, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylim(0, 1.05)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison', fontweight='bold', fontsize=11)
    for bar, val in zip(bars, acc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.2%}', ha='center', va='bottom',
                 fontweight='bold')
    plt.tight_layout()
    elements.append(add_figure(fig2, width=400, height=180))
    elements.append(PageBreak())

    # Page 3
    elements.append(add_logo_header(doc, styles, "DETAILED EVALUATION"))
    elements.append(Spacer(1, 8))

    block = []
    block.append(add_heading("Confusion Matrix"))
    fig3 = plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred_stack), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Stacking Model', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    block.append(add_figure(fig3, width=300, height=200))
    elements.append(KeepTogether(block))
    elements.append(Spacer(1, 10))

    block2 = []
    block2.append(add_heading("ROC Curve"))
    fig4 = plt.figure(figsize=(4, 3))
    fpr, tpr, _ = roc_curve(y_test, y_proba_stack)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Stacking (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    block2.append(add_figure(fig4, width=300, height=200))
    elements.append(KeepTogether(block2))
    elements.append(Spacer(1, 10))

    block3 = []
    block3.append(add_heading("Precision-Recall Curve"))
    fig5 = plt.figure(figsize=(4, 3))
    prec, rec, _ = precision_recall_curve(y_test, y_proba_stack)
    plt.plot(rec, prec, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.tight_layout()
    block3.append(add_figure(fig5, width=300, height=200))
    elements.append(KeepTogether(block3))
    elements.append(Spacer(1, 12))

    elements.append(add_heading("Classification Report"))
    report = classification_report(y_test, y_pred_stack, target_names=['Normal', 'Fraud'], output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    report_data = [[''] + list(report_df.columns)] + [[idx] + [f"{v:.3f}" if isinstance(v, float) else v for v in row]
                                                      for idx, row in report_df.iterrows()]
    for i in range(1, len(report_data)):
        for j in range(1, len(report_data[0])):
            if isinstance(report_data[i][j], str) and '.' in report_data[i][j]:
                report_data[i][j] = f"{float(report_data[i][j]):.3f}"
    report_table = Table(report_data, colWidths=[80, 70, 70, 70, 70])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(report_table)
    elements.append(PageBreak())

    # Page 4
    elements.append(add_logo_header(doc, styles, "GLOBAL EXPLANATION (SHAP)"))
    elements.append(Spacer(1, 8))

    block4 = []
    block4.append(add_heading("SHAP Beeswarm Plot"))
    fig6 = plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    block4.append(add_figure(fig6, width=450, height=280))
    elements.append(KeepTogether(block4))
    elements.append(Spacer(1, 10))

    block5 = []
    block5.append(add_heading("SHAP Feature Importance (Bar)"))
    fig7 = plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    block5.append(add_figure(fig7, width=450, height=280))
    elements.append(KeepTogether(block5))
    elements.append(PageBreak())

    # Page 5
    elements.append(add_logo_header(doc, styles, "LOCAL EXPLANATION"))
    elements.append(Spacer(1, 8))

    block6 = []
    block6.append(add_heading("LIME Explanation (Sample Instance)"))
    exp = lime_explainer.explain_instance(X_test.iloc[0].values, stack.predict_proba, num_features=8)
    fig8 = exp.as_pyplot_figure()
    fig8.set_size_inches(6, 4)
    plt.tight_layout()
    block6.append(add_figure(fig8, width=450, height=280))
    elements.append(KeepTogether(block6))
    elements.append(Spacer(1, 10))

    block7 = []
    block7.append(add_heading("SHAP Waterfall Plot (Same Instance)"))
    explainer_tree = shap.TreeExplainer(rf)
    x_instance = X_test.iloc[[0]]
    shap_exp = explainer_tree(x_instance)
    if len(shap_exp.shape) == 3:
        values = shap_exp[0, :, 1]
        base = explainer_tree.expected_value[1] if isinstance(explainer_tree.expected_value,
                                                              list) else explainer_tree.expected_value
    else:
        values = shap_exp[0]
        base = explainer_tree.expected_value
    exp_obj = shap.Explanation(values=values, base_values=base, data=x_instance.iloc[0].values,
                               feature_names=feature_names)
    fig9 = plt.figure(figsize=(7, 5))
    shap.plots.waterfall(exp_obj, max_display=10, show=False)
    plt.tight_layout()
    block7.append(add_figure(fig9, width=450, height=300))
    elements.append(KeepTogether(block7))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ------------------- Prediction PDF (Big4 style) -------------------
def generate_prediction_pdf(input_data, pred_stack, prob_stack, prob_rf, prob_gb, lower, upper, importance,
                            forecast_df=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=25, rightMargin=25, topMargin=20,
                            bottomMargin=20)
    styles = getSampleStyleSheet()
    elements = []

    def add_heading(text, level=2):
        size = 12 if level == 2 else 14
        para = Paragraph(f"<font size={size}><b>{text}</b></font>", styles['Normal'])
        para.spaceAfter = 6  # space after heading
        para.spaceBefore = 4  # space before heading
        return para

    def add_figure(fig, width=480, height=220):
        img_data = BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        img_data.seek(0)
        return Image(img_data, width=width, height=height)

    # ---------- PAGE 1 (with logo header) ----------
    elements.append(add_logo_header(doc, styles, "CONFIDENTIAL FRAUD RISK AUDIT REPORT"))
    elements.append(Spacer(1, 6))  # small gap after header

    # -------- SUBTITLE --------
    elements.append(Paragraph(
        "<b><font size=12>Fraud Detection & Risk Intelligence System</font></b>",
        styles['Heading2']
    ))
    elements.append(Spacer(1, 10))
    meta_data = [
        ["Report Detail", "Information"],
        ["Client Name", "ABC Corporation"],
        ["Report ID", f"FRAUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"],
        ["Generated On", datetime.now().strftime('%Y-%m-%d')],
        ["Generated Time", datetime.now().strftime('%H:%M:%S')],
        ["Prepared By", "AI Fraud Detection Engine"],
        ["Model Used", "Stacking (RF + GB + Logistic Regression)"],
    ]

    meta_table = Table(meta_data, colWidths=[200, 320])

    meta_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

        # Body
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),

        # Left column highlight
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),

        # Alignment
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    elements.append(add_heading("Executive Summary"))
    if pred_stack == 1:
        summary = f"""
        The analysis indicates a <b>HIGH probability of financial fraud</b> with a predicted score of 
        <b>{prob_stack * 100:.2f}%</b>.<br/>
        Multiple abnormal financial patterns suggest potential manipulation, requiring <b>immediate investigation</b>.
        """
    else:
        summary = f"""
        The company is classified as <b>LOW fraud risk</b> with a predicted probability of 
        <b>{prob_stack * 100:.2f}%</b>.<br/>
        Financial indicators appear stable, though <b>routine monitoring is recommended</b>.
        """
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(add_heading("Risk Classification"))
    risk_level = "HIGH" if prob_stack > 0.7 else "MEDIUM" if prob_stack > 0.4 else "LOW"
    risk_color = colors.red if risk_level == "HIGH" else colors.orange if risk_level == "MEDIUM" else colors.green
    risk_table = Table([
        ["Metric", "Value"],
        ["Fraud Probability", f"{prob_stack * 100:.2f}%"],
        ["Risk Level", risk_level],
        ["Confidence Interval", f"{lower:.2%} - {upper:.2%}"]
    ], colWidths=[doc.width / 2 - 20, doc.width / 2 - 20])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (1, 2), (1, 2), risk_color),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 10))

    elements.append(add_heading("Model Consensus"))
    model_table = Table([
        ["Model", "Fraud Probability"],
        ["Random Forest", f"{prob_rf:.2%}"],
        ["Gradient Boosting", f"{prob_gb:.2%}"],
        ["Stacking Model", f"{prob_stack:.2%}"]
    ], colWidths=[doc.width / 2 - 20, doc.width / 2 - 20])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(model_table)
    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph("<b>Insight:</b> All models show consistent predictions, indicating strong agreement.",
                  styles['Normal']))
    elements.append(PageBreak())

    # =========================================================
    # PAGE 2 (BIG TABLE)
    # =========================================================
    elements.append(Paragraph("<b>Key Financial Indicators</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))

    table_data = [["Feature", "Value"]]
    for k, v in list(input_data.items())[:20]:
        table_data.append([k, str(v)])

    table = Table(table_data, colWidths=[250, 250])

    table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 11),  # 👈 bigger text
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
    ]))

    elements.append(table)

    elements.append(PageBreak())

    # =========================================================
    # PAGE 3 (GRAPHS)
    # =========================================================

    # ---- Graph 1: Model Comparison ----
    fig1 = plt.figure()
    plt.bar(["RF", "GB", "STACK"], [prob_rf, prob_gb, prob_stack])
    plt.title("Model Comparison")

    img1 = BytesIO()
    plt.savefig(img1, format='png')
    plt.close()
    img1.seek(0)

    elements.append(Paragraph("<b>Model Comparison</b>", styles['Heading2']))
    elements.append(Image(img1, width=400, height=200))

    # ---- Graph 2: Top Risk Drivers ----
    fig2 = plt.figure()
    top = importance.head(10)
    plt.barh(top["Feature"], top["Impact"])
    plt.title("Top Risk Drivers")

    img2 = BytesIO()
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)

    elements.append(Paragraph("<b>Top Risk Drivers</b>", styles['Heading2']))
    elements.append(Image(img2, width=400, height=200))

    elements.append(PageBreak())

    # =========================================================
    # PAGE 4
    # =========================================================

    # ---- Graph 3 ----
    fig3 = plt.figure()
    plt.hist(y_proba_stack[y_test == 0], alpha=0.5, label='Normal')
    plt.hist(y_proba_stack[y_test == 1], alpha=0.5, label='Fraud')
    plt.legend()
    plt.title("Fraud Probability Distribution")

    img3 = BytesIO()
    plt.savefig(img3, format='png')
    plt.close()
    img3.seek(0)

    elements.append(Paragraph("<b>Fraud Probability Distribution</b>", styles['Heading2']))
    elements.append(Image(img3, width=400, height=200))

    # ---- Graph 4 (Forecast if exists) ----
    if 'year' in feature_names:
        fig4 = plt.figure()
        plt.plot(all_years, all_probs, marker='o')
        plt.title("Fraud Risk Forecast")

        img4 = BytesIO()
        plt.savefig(img4, format='png')
        plt.close()
        img4.seek(0)

        elements.append(Paragraph("<b>Fraud Risk Forecast</b>", styles['Heading2']))
        elements.append(Image(img4, width=400, height=200))

    # ---------- PAGE 5 (NO HEADER) ----------
    elements.append(Paragraph("<b>Conclusion</b>", styles['Heading2']))

    if pred_stack == 1:
        conclusion = """
            <b>High fraud risk detected.</b> Financial inconsistencies such as abnormal ratios,
            negative cash flows, and irregular patterns indicate potential manipulation.
            """

        # Add bullet points for high risk cases
        bullets = [
            "<b>Immediate actions required:</b>",
            "• Launch a full forensic audit within 7 days",
            "• Review all transactions flagged by the model (see Top Risk Drivers)",
            "• Escalate to the Audit Committee and external regulators",
            "• Freeze any suspicious accounts or pending payments",
            "",
            "<b>Key indicators observed:</b>",
            "• Negative operating cash flow despite reported profits",
            "• Unusually high debt_to_equity ratio",
            "• Mismatch between net profit and cash flow from operations",
            "",
            "<b>Recommended follow up:</b>",
            "• Reconsider vendor contracts and related party transactions",
            "• Strengthen internal controls over financial reporting",
            "• Schedule a follow up risk assessment in 30 days",
        ]
    else:
        conclusion = """
            <b>No significant fraud indicators detected.</b> Financials appear stable and consistent with industry norms.
            """

        bullets = [
            "<b>Routine monitoring recommended:</b>",
            "• Continue quarterly financial reviews",
            "• Monitor the Top Risk Drivers for any future changes",
            "• Perform random sample testing of high‑value transactions",
            "",
            "<b>Best practices to maintain low risk:</b>",
            "• Keep debt‑to‑equity ratio below 2.0",
            "• Maintain positive operating cash flow",
            "• Update fraud risk score annually",
        ]

    # Add conclusion paragraph
    elements.append(Paragraph(conclusion, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Add bullet points as a single paragraph with <br/> line breaks
    bullet_text = "<br/>".join(bullets)
    elements.append(Paragraph(bullet_text, styles['Normal']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "<i>This report was automatically generated by the FinGuard AI Risk Intelligence Platform.</i>",
        styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ------------------- Sidebar PDF Download -------------------
st.sidebar.markdown("---")
if st.sidebar.button("📄 Download Full Analysis PDF"):
    with st.spinner("Generating comprehensive report..."):
        pdf_buf = generate_full_pdf()
        st.sidebar.download_button("📥 Download Report", pdf_buf,
                                   f"fraud_analysis_{datetime.now():%Y%m%d_%H%M%S}.pdf", "application/pdf")

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout", key="logout_btn"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
if st.session_state.get('logout', False):
    st.stop()


# ------------------- Tabs -------------------
tabs = st.tabs(["🏠 Home", "📋 Data", "📊 Comparison", "🔍 Evaluation", "🧠 XAI", "🎯 Prediction"])

# ==================== HOME TAB ====================
with tabs[0]:
    st.header("📊 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred_stack):.2%}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred_stack):.2%}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred_stack):.2%}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred_stack):.2%}")

    st.subheader("📁 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Fraud Cases", int(df['anomaly_label'].sum()))

    st.subheader("📊 Fraud Distribution")
    fraud_counts = df['anomaly_label'].value_counts().reset_index()
    fraud_counts.columns = ['Class', 'Count']
    fraud_counts['Class'] = fraud_counts['Class'].map({0: 'Normal', 1: 'Fraud'})
    fig = px.bar(fraud_counts, x='Class', y='Count', text='Count', color='Class')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# ==================== DATA TAB ====================
with tabs[1]:
    st.header("📋 Data Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("🔎 Column Types")
    st.dataframe(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)}), use_container_width=True)

    st.subheader("❗ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}))
    else:
        st.success("No missing values found")

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("🎯 Target Distribution")
    target_counts = df['anomaly_label'].value_counts().reset_index()
    target_counts.columns = ['Class', 'Count']
    target_counts['Class'] = target_counts['Class'].map({0: 'Normal', 1: 'Fraud'})
    fig = px.bar(target_counts, x='Class', y='Count', text='Count', color='Class')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Explore Column")
    selected_col = st.selectbox("Select Column", df.columns)
    st.write(df[selected_col].value_counts().head(10))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("📊 Distribution Plot")
    selected_num = st.selectbox("Select Numeric Column", numeric_cols)
    fig = px.histogram(df, x=selected_num)
    st.plotly_chart(fig, use_container_width=True)

# ==================== COMPARISON TAB ====================
with tabs[2]:
    st.header("Model Performance Comparison")
    results = {
        'Model': ['Random Forest', 'Gradient Boosting', 'Proposed Model'],
        'Accuracy': [accuracy_score(y_test, y_pred_rf) * 100, accuracy_score(y_test, y_pred_gb) * 100,
                     accuracy_score(y_test, y_pred_stack) * 100],
        'Precision': [precision_score(y_test, y_pred_rf, zero_division=0) * 100,
                      precision_score(y_test, y_pred_gb, zero_division=0) * 100,
                      precision_score(y_test, y_pred_stack, zero_division=0) * 100],
        'Recall': [recall_score(y_test, y_pred_rf) * 100, recall_score(y_test, y_pred_gb) * 100,
                   recall_score(y_test, y_pred_stack) * 100],
        'F1-Score': [f1_score(y_test, y_pred_rf) * 100, f1_score(y_test, y_pred_gb) * 100,
                     f1_score(y_test, y_pred_stack) * 100]
    }
    res_df = pd.DataFrame(results).round(2)
    st.dataframe(res_df, use_container_width=True)

    st.subheader("Performance Comparison Chart")
    df_melted = res_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(11, 7))
    custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette=custom_palette, ax=ax)
    plt.title("Performance Comparison", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Score (%)")
    plt.ylim(0, 115)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, fontsize=11)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=4, rotation=90, fontsize=10, fontweight='bold')
    st.pyplot(fig, use_container_width=True)

# ==================== EVALUATION TAB ====================
with tabs[3]:
    st.header("Detailed Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred_stack), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(4, 3))
        for name, proba, color in zip(['RF', 'GB', 'Proposed'],
                                      [y_proba_rf, y_proba_gb, y_proba_stack],
                                      ['#1f77b4', '#ff7f0e', '#2ca02c']):
            fpr, tpr, _ = roc_curve(y_test, proba)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})', color=color)
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, proba, color in zip(['RF', 'GB', 'Proposed'],
                                      [y_proba_rf, y_proba_gb, y_proba_stack],
                                      ['#1f77b4', '#ff7f0e', '#2ca02c']):
            prec, rec, _ = precision_recall_curve(y_test, proba)
            ax.plot(rec, prec, label=name, color=color)
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
        st.pyplot(fig)
    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred_stack, target_names=['Normal', 'Fraud'], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance (Random Forest)")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        fig, ax = plt.subplots(figsize=(5, 6))
        sns.barplot(x=top_importances, y=top_features, palette='viridis', ax=ax)
        ax.set_title("Top 15 Most Important Features")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
    with col2:
        st.subheader("Predicted Probability Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.linspace(0, 1, 30)
        ax.hist(y_proba_stack[y_test == 0], bins=bins, alpha=0.6, label='Actual Normal', color='green',
                edgecolor='black')
        ax.hist(y_proba_stack[y_test == 1], bins=bins, alpha=0.6, label='Actual Fraud', color='red', edgecolor='black')
        ax.set_title('Distribution of Predicted Fraud Probabilities')
        ax.set_xlabel('Predicted Probability of Fraud')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        prob_true, prob_pred = calibration_curve(y_test, y_proba_stack, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Proposed Model')
        ax.plot([0, 1], [0, 1], '--', label='Perfectly Calibrated')
        ax.set_title("Calibration Curve")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns.")

# ==================== XAI TAB ====================
with tabs[4]:
    st.header("🧠 Explainable AI (SHAP & LIME)")
    st.subheader("🌍 Global Explanation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SHAP Beeswarm Plot**")
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Beeswarm error: {e}")
    with col2:
        st.markdown("**SHAP Feature Importance (Bar)**")
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Bar error: {e}")

    st.subheader("📊 Feature Analysis")
    selected_feature = st.selectbox("Select Feature", feature_names, key="feat_xai")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SHAP Dependence Plot**")
        try:
            feature_idx = feature_names.index(selected_feature)
            fig, ax = plt.subplots(figsize=(6, 5))
            shap_vals_class1 = shap_values if len(shap_values.shape) == 2 else shap_values[:, :, 1]
            shap.dependence_plot(feature_idx, shap_vals_class1, shap_sample.values,
                                 feature_names=feature_names, ax=ax, show=False)
            plt.title(f"SHAP Dependence Plot for {selected_feature}")
            st.pyplot(fig)
        except Exception:
            fig, ax = plt.subplots(figsize=(6, 5))
            feature_vals = X_test[selected_feature].values
            ax.scatter(feature_vals, y_proba_stack, alpha=0.5, s=10)
            z = np.polyfit(feature_vals, y_proba_stack, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(feature_vals), p(np.sort(feature_vals)), "r--", label="Trend")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Fraud Probability")
            ax.set_title(f"{selected_feature} vs Fraud Probability")
            ax.legend()
            st.pyplot(fig)
    with col2:
        st.markdown("**Partial Dependence Plot (PDP)**")
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(
                stack,
                X_train_res.sample(min(300, len(X_train_res)), random_state=42),
                [selected_feature],
                ax=ax
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"PDP error: {e}")

    st.subheader("🔍 Local Explanation")
    fraud_indices = np.where(y_test == 1)[0]
    if len(fraud_indices) > 0:
        instance_idx = st.selectbox("Select Fraud Instance", fraud_indices, key="xai_instance")
    else:
        instance_idx = 0
        st.warning("No fraud case found. Showing first instance.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**LIME Explanation**")
        try:
            exp = lime_explainer.explain_instance(X_test.iloc[instance_idx].values, stack.predict_proba,
                                                  num_features=10)
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"LIME error: {e}")
    with col2:
        st.markdown("**SHAP Waterfall Plot**")
        try:
            explainer_tree = shap.TreeExplainer(rf)
            x_instance = X_test.iloc[[instance_idx]]
            shap_exp = explainer_tree(x_instance)
            if len(shap_exp.shape) == 3:
                exp_for_class1 = shap_exp[0, :, 1]
                base_val = explainer_tree.expected_value[1] if isinstance(explainer_tree.expected_value,
                                                                          list) else explainer_tree.expected_value
            else:
                exp_for_class1 = shap_exp[0]
                base_val = explainer_tree.expected_value
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.waterfall_plot(
                shap.Explanation(values=exp_for_class1,
                                 base_values=base_val,
                                 data=x_instance.iloc[0].values,
                                 feature_names=feature_names),
                max_display=10, show=False
            )
            st.pyplot(fig)
        except Exception:
            fig, ax = plt.subplots(figsize=(6, 5))
            importances = rf.feature_importances_
            idx = np.argsort(importances)[-10:]
            ax.barh([feature_names[i] for i in idx], importances[idx])
            ax.set_title("Top Feature Importance (Fallback)")
            st.pyplot(fig)

    st.subheader("📊 Top SHAP Features")
    try:
        shap_2d = shap_values if len(shap_values.shape) == 2 else shap_values[:, :, 1]
        mean_abs = np.abs(shap_2d).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": feature_names, "Importance": mean_abs}).sort_values(by="Importance",
                                                                                               ascending=False)
        st.dataframe(shap_df.head(15), width='stretch')
    except Exception as e:
        st.error(f"SHAP table error: {e}")

# ==================== PREDICTION TAB ====================
with tabs[5]:
    st.header("🎯 Smart Fraud Prediction System")
    st.markdown("### 📌 Enter Company Financial Details")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if 'auto_input' not in st.session_state:
        sample = df[df['anomaly_label'] == 1].sample(1, random_state=42)
        st.session_state['auto_input'] = sample.drop(columns=['anomaly_label']).iloc[0].to_dict()

    input_data = {}
    cols = st.columns(3)
    auto_data = st.session_state.get('auto_input', {})

    for i, col in enumerate(feature_names):
        with cols[i % 3]:
            default_val = auto_data.get(col, None)
            if col in categorical_cols:
                options = sorted(df[col].dropna().unique())
                index = options.index(default_val) if default_val in options else 0
                input_data[col] = st.selectbox(col, options, index=index, key=f"input_{col}")
            else:
                input_data[col] = st.number_input(
                    col,
                    value=float(default_val) if default_val is not None else float(X_train_res[col].mean()),
                    key=f"input_{col}"
                )

    if st.button("🚀 Run Prediction", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))
        input_scaled = scaler.transform(input_df)

        pred_rf = rf.predict(input_scaled)[0]
        pred_gb = gb.predict(input_scaled)[0]
        pred_stack = stack.predict(input_scaled)[0]
        prob_rf = rf.predict_proba(input_scaled)[0][1]
        prob_gb = gb.predict_proba(input_scaled)[0][1]
        prob_stack = stack.predict_proba(input_scaled)[0][1]
        prob = prob_stack

        st.subheader("📊 Prediction Result")
        if pred_stack == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Normal")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Fraud Risk (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        voting_df = pd.DataFrame({
            "Model": ["RF", "GB", "STACK"],
            "Probability": [prob_rf, prob_gb, prob_stack]
        })
        fig_bar = px.bar(voting_df, x="Model", y="Probability", text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)

        probs = np.array([prob_rf, prob_gb, prob_stack])
        mean = probs.mean()
        std = probs.std()
        lower = max(0, mean - 1.96 * std)
        upper = min(1, mean + 1.96 * std)
        st.info(f"Confidence Interval: [{lower:.2%}, {upper:.2%}]")

        try:
            explainer = get_shap_explainer()
            shap_vals = explainer.shap_values(input_scaled)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_vals = np.array(shap_vals)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            if shap_vals.ndim == 2:
                shap_vals = shap_vals[0]
            shap_vals = shap_vals.flatten()
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Impact": shap_vals
            }).sort_values(by="Impact", key=np.abs, ascending=False).head(10)
        except:
            importances = rf.feature_importances_
            idx = np.argsort(importances)[::-1][:10]
            importance = pd.DataFrame({
                "Feature": [feature_names[i] for i in idx],
                "Impact": importances[idx]
            })

        fig_shap = px.bar(importance, x="Impact", y="Feature", orientation='h')
        st.plotly_chart(fig_shap, use_container_width=True)

        # Forecast
        forecast_df = None
        if 'year' in feature_names:
            st.subheader("🔮 Fraud Risk Forecast")
            years = sorted(df['year'].unique())
            last_year = max(years)
            probs_hist = []
            for y in years:
                temp = input_df.copy()
                temp['year'] = y
                p = stack.predict_proba(scaler.transform(temp))[0][1]
                probs_hist.append(p)
            future_years = [last_year + i for i in range(1, 4)]
            probs_future = []
            for y in future_years:
                temp = input_df.copy()
                temp['year'] = y
                p = stack.predict_proba(scaler.transform(temp))[0][1]
                probs_future.append(p)
            all_years = years + future_years
            all_probs = probs_hist + probs_future
            forecast_df = pd.DataFrame({
                "Year": all_years,
                "Fraud Probability": all_probs
            })
            forecast_df["Upper"] = np.clip(forecast_df["Fraud Probability"] + std, 0, 1)
            forecast_df["Lower"] = np.clip(forecast_df["Fraud Probability"] - std, 0, 1)
            fig_forecast = px.line(forecast_df, x="Year", y="Fraud Probability", markers=True)
            fig_forecast.add_traces([
                go.Scatter(x=forecast_df["Year"], y=forecast_df["Upper"], line=dict(width=0), showlegend=False),
                go.Scatter(x=forecast_df["Year"], y=forecast_df["Lower"], fill='tonexty', name='Confidence')
            ])
            st.plotly_chart(fig_forecast, use_container_width=True)

        pdf_pred = generate_prediction_pdf(input_data, pred_stack, prob_stack, prob_rf, prob_gb, lower, upper,
                                           importance, forecast_df)
        st.download_button("📥 Download Prediction Report", pdf_pred,
                           f"Prediction_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf", "application/pdf")