def interpret_model_predictions(model, X, feature_names):
    import shap

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X)
    
    # Calculate SHAP values
    shap_values = explainer(X)

    # Plot summary of SHAP values
    shap.summary_plot(shap_values, X, feature_names=feature_names)

def plot_feature_importance(model, feature_names):
    import matplotlib.pyplot as plt
    import numpy as np

    # Get feature importance from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature importances.")

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()