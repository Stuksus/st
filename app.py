
import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load the default Breast Cancer dataset and use a subset of data for faster training
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names
df = pd.DataFrame(X, columns=features)
df['target'] = y

# Reduce dataset size for optimization
df = df.sample(frac=0.3, random_state=42)  # Use only 30% of the dataset

# Initialize session state to store model history and parameters
if 'model_history' not in st.session_state:
    st.session_state['model_history'] = []
if 'model_parameters' not in st.session_state:
    st.session_state['model_parameters'] = {}

# Add DRPA title separately and make it bigger
st.markdown("<h1 style='text-align: left; font-size: 4rem;'>DRPA</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left;'>Попробуйте сейчас</h2>", unsafe_allow_html=True)

# Sidebar for statistics table and feature sample selection
st.sidebar.title("DRPA")

# Displaying stored model results as a table in the sidebar
st.sidebar.subheader("Статистика")
if st.session_state['model_history']:
    history_df = pd.DataFrame(st.session_state['model_history'], columns=["Версия", "Оценка", "Метрика"])
    st.sidebar.table(history_df)
else:
    st.sidebar.write("Нет ранее обученных моделей.")

# Sidebar for feature sample selection
st.sidebar.subheader("Выбор признаков")
selected_feature = st.sidebar.selectbox("Выберите признак", df.columns[:-1])
if selected_feature:
    st.sidebar.write("Образцы признака")
    st.sidebar.table(df[[selected_feature]].sample(5))

# Right-side UI for input and model training
st.subheader("Выберите алгоритм:")
algorithm_choice = st.radio("Выберите алгоритм", ["Random Forest", "Decision Tree", "Logistic Regression", "LightGBM"])

# Choose features with multi-select dropdown (limit to 5 features manually)
st.subheader("Выберите признаки (до 5):")
selected_features = st.multiselect("Выберите признаки", df.columns[:-1])

if len(selected_features) > 5:
    st.warning("Вы можете выбрать не более 5 признаков.")
    selected_features = selected_features[:5]

# Metric selection as radio buttons (only one can be selected)
st.subheader("Выберите основную метрику (только одну):")
metric = st.radio("Выберите метрику", ["F1-score", "Accuracy", "Recall"])

# Button to train
if st.button("Запустить обучение"):

    # Ensure at least 2 features are selected
    if len(selected_features) < 2:
        st.warning("Пожалуйста, выберите как минимум 2 признака.")
    else:
        # Prepare the data for the selected features and include the target column
        X = df[selected_features]
        y = df['target']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner('Обучение модели...'):
            if algorithm_choice == "LightGBM":
                model = LGBMClassifier(n_estimators=50, max_depth=5)  # Limit estimators and depth for fast training

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

            elif algorithm_choice == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, max_depth=5)  # Limit estimators and depth

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

            elif algorithm_choice == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(max_depth=5)  # Limit tree depth

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

            elif algorithm_choice == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=200)

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

            # Calculate the chosen metric
            if metric == "F1-score":
                score = f1_score(y_test, y_pred, average='weighted')
            elif metric == "Accuracy":
                score = accuracy_score(y_test, y_pred)
            else:
                score = recall_score(y_test, y_pred, average='weighted')

            # Display the score
            st.write(f"{metric}: {score:.2f}")

            # Save the result and parameters to session state
            model_name = f"{algorithm_choice}_model"
            st.session_state['model_history'].append((model_name, score, metric))
            st.session_state['model_parameters'][model_name] = {"Algorithm": algorithm_choice}

            # PCA for 2D plot
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)

            # Plotting the decision boundary
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))

            Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)

            # Plot the decision boundary with colors
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
            plt.title(f"Граница решений для {algorithm_choice}")
            plt.xlabel('PCA Компонента 1')
            plt.ylabel('PCA Компонента 2')
            st.pyplot(plt.gcf())
