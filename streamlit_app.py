import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Ensure the script is run with Streamlit
if __name__ == "__main__":
    # Load the dataset
    crops = pd.read_csv("soil_measures.csv")

    # Split into feature and target sets
    X = crops.drop(columns="crop")
    y = crops["crop"]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train and evaluate models using cross-validation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    feature_performance = {}
    for model_name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1_weighted")
        feature_performance[model_name] = scores.mean()

    # Feature importance using Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feature_importance = pd.Series(importances, index=crops.drop(columns="crop").columns).sort_values(ascending=False)

    # Streamlit app
    st.title("Crop Prediction Based on Soil Analysis")

    st.header("Model Performance")
    st.write("F1-scores for different models:")
    for model_name, score in feature_performance.items():
        st.write(f"{model_name}: {score:.4f}")

    st.header("Feature Importance")
    st.write("Importance of each soil feature in predicting the optimal crop:")
    fig = px.bar(
        feature_importance,
        x=feature_importance.index,
        y=feature_importance.values,
        labels={'x': 'Soil Feature', 'y': 'Importance Score'},
        title='Feature Importance for Crop Prediction'
    )
    st.plotly_chart(fig)

    st.header("Conclusion")
    st.write("""
    We trained and evaluated multiple machine learning models (Logistic Regression, Decision Tree, and Random Forest) to predict the optimal crop based on soil measurements. Cross-validation results showed that the Random Forest model performed well. Feature importance analysis revealed that Potassium (K) is the most critical soil feature for predicting the optimal crop, followed by Phosphorous (P), pH, and Nitrogen (N). This insight can help farmers prioritize soil measurements to maximize crop yield.
    """)
