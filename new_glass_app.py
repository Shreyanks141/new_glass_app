import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
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
st.title("Glass Type prediction Web app")
st.sidebar.title("Glass Type prediction Web app")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Glass Type Data set")
    st.dataframe(glass_df)
st.sidebar.subheader("Scatter Plot")

features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)

for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()

st.sidebar.subheader("Visual selector")
plot_type = st.sidebar.multiselect("Select type of plot",('Histogram','Boxplot','Count Plot','Pychart','Correlation Heatmap','Pair Plot'))
if  "Histogram"  in plot_type:
    st.subheader("Histogram")
    hist_features = st.sidebar.multiselect("Select features to create histograms:", 
                                                ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    for feature in hist_features:
        plt.figure(figsize = (12, 6))
        plt.title(f"Histogram for {feature}")
        plt.hist(glass_df[feature], bins = 'sturges', edgecolor = 'black')
        st.pyplot() 
if  "Boxplot"  in plot_type:
    st.subheader("Box Plot")
    box_plot_cols = st.sidebar.multiselect("Select the columns to create box plots:",
                                                ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
    for col in box_plot_cols:
        plt.figure(figsize = (12, 2))
        plt.title(f"Box plot for {col}")
        sns.boxplot(glass_df[col])
        st.pyplot() 
if "Count Plot" in plot_type:
    st.subheader("Count Plot")
    plt.figure(figsize = (12, 2))
    sns.countplot(glass_df["GlassType"])
    st.pyplot() 
if "Pychart" in plot_type:
    st.subheader("Pychart")
    glass_count = glass_df["GlassType"].value_counts()
    plt.figure(figsize = (12, 2))
    plt.pie( glass_count,labels = glass_count.index,autopct = "%1.2f%%",explode = np.linspace(0.01,0.06,6))
    st.pyplot()
if "Correlation Heatmap" in plot_type:
    st.subheader("Correlation Heatmap")
    plt.figure(figsize = (12, 10),dpi = 95)
    sns.heatmap(glass_df.corr(),annot = True)
    st.pyplot()
if "Pair Plot" in plot_type:
    st.subheader("Pair Plot")
    plt.figure(figsize = (12, 2),dpi = 95)
    sns.pairplot(glass_df)
    st.pyplot()


st.sidebar.subheader("Select the features values")
ri = st.sidebar.slider("RI", float(glass_df["RI"].min()), float(glass_df["RI"].max()))
na = st.sidebar.slider("Na", float(glass_df["Na"].min()), float(glass_df["Na"].max()))
mg = st.sidebar.slider("Mg", float(glass_df["Mg"].min()), float(glass_df["Mg"].max()))
al = st.sidebar.slider("Al", float(glass_df["Al"].min()), float(glass_df["Al"].max()))
si = st.sidebar.slider("Si", float(glass_df["Si"].min()), float(glass_df["Si"].max()))
k = st.sidebar.slider("K", float(glass_df["K"].min()), float(glass_df["K"].max()))
ca = st.sidebar.slider("Ca", float(glass_df["Ca"].min()), float(glass_df["Ca"].max()))
ba = st.sidebar.slider("Ba", float(glass_df["Ba"].min()), float(glass_df["Ba"].max()))
fe = st.sidebar.slider("Fe", float(glass_df["Fe"].min()), float(glass_df["Fe"].max()))


st.sidebar.subheader("Classifier")
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if classifier == "Support Vector Machine":
    c = st.sidebar.number_input("C (ERROR Rate)",1,100,step = 1)
    kernel_input = st.sidebar.radio("Kernel",('linear',"poly","rbf"))
    gamma_input = st.sidebar.number_input("Gamma Input",1,100,step = 1)

    if st.sidebar.button("Classify"):
        st.subheader("svc_model")
        svc_model = SVC(kernel = kernel_input,gamma = gamma_input,C = c)
        svc_model.fit(X_train, y_train)
        score_svc = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("Type Of Glass Predicted is",glass_type)
        st.write("Accuracy_score",score_svc)
        plot_confusion_matrix(svc_model,X_test,y_test)
        st.pyplot()    

if classifier == "Random Forest Classifier":
    n_estimators_input = st.sidebar.number_input("Number of trees",100,5000,step = 10)
    max_depth_input = st.sidebar.number_input("Depth of trees",1,100,step = 1)

    if st.sidebar.button("Classify"):
        st.subheader("rf_clf")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input,max_depth = max_depth_input,n_jobs  = -1)
        rf_clf.fit(X_train, y_train)
        score_rf_clf = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("Type Of Glass Predicted is",glass_type)
        st.write("Accuracy_score",score_rf_clf)
        plot_confusion_matrix(rf_clf,X_test,y_test)
        st.pyplot()
