import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_random_forest(df_features):
    X = df_features.drop(columns=["class"])
    y = df_features["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=5,
        min_samples_split=5,
        max_features=0.2,
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    importances = rf.feature_importances_
    feature_names = X_train_scaled.columns
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title("Feature Importances")
    plt.show()

    return rf, scaler
