import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def clean_data(df):
    # Handle missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df['gender_Female'] = (df['gender'] == 'Female').astype(int)
    df['gender_Male'] = (df['gender'] == 'Male').astype(int)
    df = df.drop(columns=['gender'])

    df = pd.get_dummies(df, drop_first=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            known = df[df[col].notna()]
            unknown = df[df[col].isna()]

            model = RandomForestRegressor(random_state=42)
            model.fit(known.drop(columns=[col]), known[col])
            df.loc[df[col].isna(), col] = model.predict(unknown.drop(columns=[col]))

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    features_to_scale = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df