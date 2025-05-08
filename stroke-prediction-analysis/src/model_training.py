def train_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    print(X_train.shape)
    print(y_train.shape)
    print(X_train.head())
    print(y_train.value_counts())
    print("Initializing Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)  
    print("Fitting the model...")
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    from sklearn.svm import SVC
    model = SVC()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model

def train_models(X_train, y_train, model_type):
    print(f"Training model: {model_type}")
    if model_type == 'logistic':
        return train_logistic_regression(X_train, y_train)
    elif model_type == 'random_forest':
        return train_random_forest(X_train, y_train)
    elif model_type == 'xgboost':
        return train_xgboost(X_train, y_train)
    elif model_type == 'svm':
        return train_svm(X_train, y_train)
    elif model_type == 'knn':
        return train_knn(X_train, y_train)
    else:
        raise ValueError("Invalid model type specified.")