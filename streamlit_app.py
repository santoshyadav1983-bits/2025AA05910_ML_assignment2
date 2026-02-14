"""
Interactive Streamlit Web Application
Classification Models on Demographics Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Classification Models Comparison Dashboard")
st.markdown("**Dataset:** Demographics.csv (Adult Income Dataset)")
st.markdown("**Objective:** Compare 6 ML models for income prediction (<=50K or >50K)")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", 
    ["🏠 Home", "📁 Dataset Overview", "🔧 Data Preprocessing", 
     "🤖 Model Training", "📈 Model Comparison"])

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    df = pd.read_csv('demographics.csv', names=column_names, skipinitialspace=True)
    df = df.replace('?', np.nan)
    
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset"""
    # Drop missing values
    df_clean = df.dropna()
    
    # Separate features and target
    X = df_clean.drop('income', axis=1)
    y = df_clean['income']
    
    # Encode target
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y)
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Encode categorical variables
    X_encoded = X.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    return X_encoded, y_encoded, label_encoder_y, label_encoders, categorical_cols, numerical_cols

@st.cache_resource
def train_all_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train all 6 models and return results"""
    results = []
    models = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    results.append(evaluate_model(lr_model, "Logistic Regression", X_train_scaled, X_test_scaled, y_train, y_test))
    models['Logistic Regression'] = lr_model
    
    # 2. Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    results.append(evaluate_model(dt_model, "Decision Tree", X_train, X_test, y_train, y_test))
    models['Decision Tree'] = dt_model
    
    # 3. KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    results.append(evaluate_model(knn_model, "K-Nearest Neighbors", X_train_scaled, X_test_scaled, y_train, y_test))
    models['K-Nearest Neighbors'] = knn_model
    
    # 4. Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    results.append(evaluate_model(nb_model, "Naive Bayes (Gaussian)", X_train_scaled, X_test_scaled, y_train, y_test))
    models['Naive Bayes (Gaussian)'] = nb_model
    
    # 5. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    results.append(evaluate_model(rf_model, "Random Forest", X_train, X_test, y_train, y_test))
    models['Random Forest'] = rf_model
    
    # 6. XGBoost
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    results.append(evaluate_model(xgb_model, "XGBoost", X_train, X_test, y_train, y_test))
    models['XGBoost'] = xgb_model
    
    return pd.DataFrame(results), models

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Evaluate a single model"""
    y_pred_test = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred_test
    
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred_test, average='weighted'),
        'Recall': recall_score(y_test, y_pred_test, average='weighted'),
        'F1': f1_score(y_test, y_pred_test, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred_test)
    }

# Load data
df = load_data()
X, y, label_encoder_y, label_encoders, categorical_cols, numerical_cols = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train models
results_df, models = train_all_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to the ML Classification Models Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Models Trained", 6)
    
    st.markdown("---")
    
    st.subheader("📋 Project Overview")
    st.write("""
    This interactive dashboard demonstrates the implementation and comparison of 6 different 
    machine learning classification models on the Adult Income Dataset (Demographics.csv).
    
    **Models Implemented:**
    1. **Logistic Regression** - Linear model for binary classification
    2. **Decision Tree** - Tree-based model with decision rules
    3. **K-Nearest Neighbors** - Instance-based learning algorithm
    4. **Naive Bayes (Gaussian)** - Probabilistic classifier
    5. **Random Forest** - Ensemble of decision trees
    6. **XGBoost** - Gradient boosting ensemble method
    
    **Features:**
    - 📊 Interactive data exploration
    - 🔧 Data preprocessing visualization
    - 🤖 Model training and evaluation
    - 📈 Performance comparison with charts
    - 🎯 Feature importance analysis
    - 🔮 Make predictions with trained models
    """)
    
    st.markdown("---")
    
    # Quick Performance Overview
    st.subheader("🏆 Quick Performance Overview")
    
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_model['Model'])
    with col2:
        st.metric("Accuracy", f"{best_model['Accuracy']:.2%}")
    with col3:
        st.metric("AUC Score", f"{best_model['AUC']:.4f}")
    with col4:
        st.metric("F1-Score", f"{best_model['F1']:.4f}")

# ============================================================================
# PAGE: DATASET OVERVIEW
# ============================================================================
elif page == "📁 Dataset Overview":
    st.header("Dataset Overview")
    
    tab1, tab2, tab3 = st.tabs(["📊 Basic Info", "📈 Statistics", "🔍 Data Quality"])
    
    with tab1:
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
            st.metric("Total Columns", len(df.columns))
        with col2:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            st.metric("Rows After Cleaning", f"{len(df.dropna()):,}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Column Types")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numerical Columns:**")
            st.write(numerical_cols)
        with col2:
            st.write("**Categorical Columns:**")
            st.write(categorical_cols)
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Target distribution
        st.subheader("Target Variable Distribution")
        target_counts = df['income'].value_counts()
        
        fig = px.pie(values=target_counts.values, names=target_counts.index, 
                     title="Income Distribution",
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Missing Values Analysis")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        title="Missing Values by Column",
                        labels={'x': 'Column', 'y': 'Missing Count'},
                        color=missing_data.values,
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected after preprocessing!")

# ============================================================================
# PAGE: DATA PREPROCESSING
# ============================================================================
elif page == "🔧 Data Preprocessing":
    st.header("Data Preprocessing Steps")
    
    st.subheader("1️⃣ Missing Value Treatment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows Before", f"{len(df):,}")
    with col2:
        st.metric("Rows After", f"{len(df.dropna()):,}")
    
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    st.write(f"**Columns with missing values:** {list(missing_counts.index)}")
    
    st.markdown("---")
    
    st.subheader("2️⃣ Feature Encoding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Categorical Features Encoded:**")
        for col in categorical_cols[:5]:
            st.write(f"- {col}")
    
    with col2:
        st.write("**Numerical Features Scaled:**")
        for col in numerical_cols[:5]:
            st.write(f"- {col}")
    
    st.markdown("---")
    
    st.subheader("3️⃣ Train-Test Split")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set", f"{len(X_train):,} (80%)")
    with col2:
        st.metric("Test Set", f"{len(X_test):,} (20%)")
    with col3:
        st.metric("Features", X_train.shape[1])
    
    # Visualize train-test split
    fig = go.Figure(data=[
        go.Bar(name='Training', x=['Count'], y=[len(X_train)], marker_color='#4ECDC4'),
        go.Bar(name='Test', x=['Count'], y=[len(X_test)], marker_color='#FF6B6B')
    ])
    fig.update_layout(title='Train-Test Split Distribution', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: MODEL TRAINING
# ============================================================================
elif page == "🤖 Model Training":
    st.header("Model Training & Evaluation")
    
    selected_model = st.selectbox("Select a Model to View Details", 
                                   results_df['Model'].tolist())
    
    model_result = results_df[results_df['Model'] == selected_model].iloc[0]
    
    # Metrics Display
    st.subheader(f"📊 {selected_model} Performance Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Accuracy", f"{model_result['Accuracy']:.4f}")
    with col2:
        st.metric("AUC", f"{model_result['AUC']:.4f}")
    with col3:
        st.metric("Precision", f"{model_result['Precision']:.4f}")
    with col4:
        st.metric("Recall", f"{model_result['Recall']:.4f}")
    with col5:
        st.metric("F1-Score", f"{model_result['F1']:.4f}")
    with col6:
        st.metric("MCC", f"{model_result['MCC']:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    model = models[selected_model]
    if selected_model in ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes (Gaussian)']:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['<=50K', '>50K'],
                    y=['<=50K', '>50K'],
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title=f'{selected_model} - Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    
    if hasattr(model, 'predict_proba'):
        if selected_model in ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes (Gaussian)']:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{selected_model} (AUC = {model_result["AUC"]:.4f})',
                                line=dict(color='#4ECDC4', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier',
                                line=dict(color='gray', width=2, dash='dash')))
        fig.update_layout(title='ROC Curve',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================
elif page == "📈 Model Comparison":
    st.header("Model Comparison")
    
    # Metrics Comparison Table
    st.subheader("📊 Performance Metrics Comparison")
    
    # Display the dataframe with formatted values
    display_df = results_df.copy()
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Metric Selection for Comparison
    st.subheader("📈 Visual Comparison")
    
    metric_to_compare = st.selectbox("Select Metric to Compare", 
                                      ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
    
    # Bar Chart
    fig = px.bar(results_df, x='Model', y=metric_to_compare,
                 title=f'{metric_to_compare} Comparison Across Models',
                 color=metric_to_compare,
                 color_continuous_scale='Viridis',
                 text_auto='.4f')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar Chart
    st.subheader("🎯 Multi-Metric Radar Chart")
    
    fig = go.Figure()
    
    for idx, row in results_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['AUC'], row['Precision'], row['Recall'], row['F1'], row['MCC']],
            theta=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="All Models - Multi-Metric Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best Model Highlight
    st.markdown("---")
    st.subheader("🏆 Best Performing Model")
    
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Model:** {best_model['Model']}")
        st.info(f"**Accuracy:** {best_model['Accuracy']:.4f}")
        st.info(f"**AUC:** {best_model['AUC']:.4f}")
    
    with col2:
        st.info(f"**Precision:** {best_model['Precision']:.4f}")
        st.info(f"**Recall:** {best_model['Recall']:.4f}")
        st.info(f"**F1-Score:** {best_model['F1']:.4f}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Classification Models Dashboard | Built with Streamlit 🎈</p>
    </div>
    """, unsafe_allow_html=True)
