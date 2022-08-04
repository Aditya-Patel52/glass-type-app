import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

@st.cache()
def prediction(model, features):
  glass_type = model.predict([features])
  glass_type = glass_type[0]
  
  if glass_type == 1: return "building windows float processed"
  elif glass_type == 2: return "building windows non float processed"
  elif glass_type == 3: return "vehicle windows float processed"
  elif glass_type == 4: return "vehicle windows non float processed"
  elif glass_type == 5: return "containers"
  elif glass_type == 6: return "tableware"
  else: return "headlamp"

st.title("Glass Type Prediction Web App")
st.sidebar.title("Glass Type Prediction Web App")

if st.sidebar.checkbox("Show raw data"):
  st.subheader("Glass Type Dataset")
  st.dataframe(glass_df)

st.sidebar.subheader("Visualisation Selector")
plot_list = st.sidebar.multiselect('Select the charts/plot:', ('Correlation heatmap', 'Line chart', 'Area chart', 'Count plot', 'Pie chart', 'Box plot'))

if 'Line chart' in plot_list:
  st.subheader('Line Chart')
  st.line_chart(glass_df)
if 'Area chart' in plot_list:
  st.subheader('Area Chart')
  st.area_chart(glass_df)

st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Correlation heatmap' in plot_list:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize=(20, 5))
  sns.heatmap(glass_df.corr(), annot=True)
  st.pyplot()
if 'Count plot' in plot_list:
  st.subheader('Count Plot')
  plt.figure(figsize=(20, 5))
  sns.countplot(glass_df['GlassType'])
  st.pyplot()
if 'Pie chart' in plot_list:
  st.subheader('Pie Chart')
  data = glass_df['GlassType'].value_counts()
  plt.figure(figsize=(20, 5))
  plt.pie(data, labels=data.index, autopct='%1.2f%%')
  st.pyplot()
if 'Box plot' in plot_list:
  st.subheader('Box Plot')
  col = st.sidebar.selectbox('Select Boxplot column', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(20, 5))
  sns.boxplot(glass_df[col])
  st.pyplot()

st.sidebar.subheader('Select your values')
ri = st.sidebar.slider('Input Ri', float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider('Input Na', float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider('Input Mg', float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider('Input Al', float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider('Input Si', float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider('Input K', float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider('Input Ca', float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider('Input Ba', float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider('Input Fe', float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader('Choose Classifier')
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyperparameters')
  c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step=1)
  kernel_input = st.sidebar.radio("Kernel", ('linear', 'rbf', 'poly'))
  gamma_value = st.sidebar.number_input("Gamma", 1, 100, step=1)
  if st.sidebar.button('Classify'):
    st.subheader("Support Vector Machine")
    svm = SVC(kernel=kernel_input, C=c_value, gamma=gamma_value)
    svm.fit(X_train, y_train)
    y_predict = svm.predict(X_test)
    accuracy = svm.score(X_train, y_train)
    glass_type = prediction(svm, [ri, na, mg, al, si, k, ca, ba, fe])
    st.write("The type of glass predicted is:", glass_type)
    st.write('Accuracy:', accuracy)
    plot_confusion_matrix(svm, X_test, y_test)
    st.pyplot()

if classifier == 'Random Forest Classifier':
  st.sidebar.subheader('Model Hyperparameters')
  estimators_input = st.sidebar.number_input('Number of trees in the forest', 1, 1000, step=10)
  max_depth_input = st.sidebar.number_input('Max depth of the tree', 1, 100, step=10)
  if st.sidebar.button('Classify'):
    st.subheader("Random Forest Classifier")
    rfc = RandomForestClassifier(n_estimators=estimators_input, max_depth=max_depth_input, n_jobs=-1)
    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    accuracy = rfc.score(X_train, y_train)
    glass_type = prediction(rfc, [ri, na, mg, al, si, k, ca, ba, fe])
    st.write("The type of glass predicted is:", glass_type)
    st.write('Accuracy:', accuracy)
    plot_confusion_matrix(rfc, X_test, y_test)
    st.pyplot() 

if classifier == 'Logistic Regression':
  st.sidebar.subheader('Model Hyperparameters')
  c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step=1)
  max_iter_input = st.sidebar.slider('Max Iterations', 10, 1000)
  if st.sidebar.button('Classify'):
    st.subheader('Logistic Regression')
    log_reg = LogisticRegression(C=c_value, max_iter=max_iter_input)
    log_reg.fit(X_train, y_train)
    y_predict = log_reg.predict(X_test)
    accuracy = log_reg.score(X_train, y_train)
    glass_type = prediction(log_reg, [ri, na, mg, al, si, k, ca, ba, fe])
    st.write("The type of glass predicted is:", glass_type)
    st.write('Accuracy:', accuracy)
    plot_confusion_matrix(log_reg, X_test, y_test)
    st.pyplot()
