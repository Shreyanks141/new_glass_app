import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)

    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}

    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]

        df.rename(columns_dict, axis = 1, inplace = True)

    return df

glass_df = load_data() 

X = glass_df.iloc[:, :-1]

y = glass_df['GlassType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()


st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")


if st.sidebar.checkbox("Show raw data"):
    st.subheader("Full Dataset")
    st.dataframe(glass_df)

st.sidebar.subheader("Scatter Plot")


st.set_option('deprecation.showPyplotGlobalUse', False)

features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                        ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()


st.sidebar.subheader("Visualisation Selector")


plot_types = st.sidebar.multiselect("Select the charts or plots:", 
                                    ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))


if 'Histogram' in plot_types:
    st.subheader("Histogram")
    columns = st.sidebar.selectbox("Select the column to create its histogram",
                                  ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

    plt.figure(figsize = (12, 6))
    plt.title(f"Histogram for {columns}")
    plt.hist(glass_df[columns], bins = 'sturges', edgecolor = 'black')
    st.pyplot()


if 'Box Plot' in plot_types:
    st.subheader("Box Plot")
    columns = st.sidebar.selectbox("Select the column to create its box plot",
                                  ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
    plt.figure(figsize = (12, 2))
    plt.title(f"Box plot for {columns}")
    sns.boxplot(glass_df[columns])
    st.pyplot()

if 'Count Plot' in plot_types:
    st.subheader("Count plot")
    sns.countplot(x = 'GlassType', data = glass_df)
    st.pyplot()
  
if 'Pie Chart' in plot_types:
    st.subheader("Pie Chart")
    pie_data = glass_df['GlassType'].value_counts()
    plt.figure(figsize = (5, 5))
    plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', 
            startangle = 30, explode = np.linspace(.06, .16, 6))
    st.pyplot()


if 'Correlation Heatmap' in plot_types:
    st.subheader("Correlation Heatmap")
    plt.figure(figsize = (10, 6))
    ax = sns.heatmap(glass_df.corr(), annot = True) 
    bottom, top = ax.get_ylim() 
    ax.set_ylim(bottom + 0.5, top - 0.5) 
    st.pyplot()

if 'Pair Plot' in plot_types:
    st.subheader("Pair Plots")
    plt.figure(figsize = (15, 15))
    sns.pairplot(glass_df)
    st.pyplot()

st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))


if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st. sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()


if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)
        
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rf_clf, X_test, y_test)
        st.pyplot()
        
if  classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    max_iter_input =  st.sidebar.number_input("Maximum iterations",10,1000,step =10)
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        lr = LogisticRegression(C = c_value,max_iter =max_iter_input)
        lr.fit(X_train,y_train)
        accuracy = lr.score(X_test, y_test)
        glass_type = prediction(lr, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(lr, X_test, y_test)
        st.pyplot()
