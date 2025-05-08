import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    # 1. Distribucija numeričkih značajki
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    for feature in numeric_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # 2. Korelacijska matrica
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # 3. Odnos značajki s ciljem (stroke)
    categorical_features = [col for col in df.columns if df[col].dtype == 'object' or len(df[col].unique()) < 10]
    categorical_features = [feature for feature in categorical_features if feature != 'stroke']  

    print(f"Categorical features: {categorical_features}")

    # Vizualiziraj odnos između kategorijskih značajki i ciljne varijable
    for feature in categorical_features:
        if feature in df.columns:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=feature, y='stroke', data=df)
            plt.title(f'{feature} vs Stroke')
            plt.show()
        else:
            print(f"Column {feature} does not exist in DataFrame.")

    # 4. Provjera neuravnoteženosti klasa
    plt.figure(figsize=(6, 4))
    sns.countplot(x='stroke', data=df)
    plt.title('Class Distribution (Stroke)')
    plt.show()

def visualize_distribution(data, column):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_correlation_matrix(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

def visualize_relationship(data, feature, target):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target, y=feature, data=data)
    plt.title(f'Relationship between {feature} and {target}')
    plt.xlabel(target)
    plt.ylabel(feature)
    plt.show()