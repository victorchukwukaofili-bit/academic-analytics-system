import joblib
import pandas as pd
import shap

def generate_student_explanation(student_idx: int = 0):
    # Load model and dataset
    model = joblib.load("model.pkl")
    df = pd.read_csv("student_data.csv")
    X = df.drop(columns=["is_at_risk"])
    
    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Get explanation for specific student instance
    student_data = X.iloc[[student_idx]]
    student_shap = shap_values[student_idx]
    
    # Extract top risk drivers
    feature_names = X.columns
    contributions = pd.DataFrame({
        "feature": feature_names,
        "value": student_data.values[0],
        "shap_value": student_shap.values
    }).sort_values(by="shap_value", ascending=True)  # lower SHAP = pushes toward risk
    
    print(f"\n--- Risk Explanation for Student Record #{student_idx} ---")
    print(f"Predicted At-Risk Probability: {model.predict_proba(student_data)[0][1]:.2%}\n")
    print("Top Influencing Factors:")
    print(contributions.to_string(index=False))

if __name__ == "__main__":
    generate_student_explanation(student_idx=5)