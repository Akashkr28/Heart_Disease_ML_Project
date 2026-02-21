import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    X = df.drop("condition", axis=1)
    y = df["condition"]

    X.fillna(X.mean(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test