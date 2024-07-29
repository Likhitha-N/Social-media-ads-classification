import streamlit as st
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Title of the application
st.title('Enhancing Social Media Marketting')

# Upload CSV file
st.sidebar.header('Upload CSV')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Function to load data
@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df
# Function to train the model
def train_model(df):
    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Model evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Displaying accuracy with customized style
    st.write('Classifier Accuracy: ', f"<span style='font-size: 24px; font-weight: bold;'>{accuracy}</span>", unsafe_allow_html=True)
    
    # Classification report
    st.write('Classification Report:')
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    # Plotting the bar graph between 'Age' and 'Purchased'
    st.subheader('Bar Graph: Age vs Purchased')
    fig, ax = plt.subplots()
    sns.barplot(x='Purchased', y='Age', data=df, hue='Purchased', palette={0: "red", 1: "green"}, ax=ax)
    st.pyplot(fig)
    st.write('<span style="font-size:24px;">0-Not Purchased</span> <span style="font-size:24px;">1-Purchased</span>', unsafe_allow_html=True)
    # Plotting the Box Plot of Estimated Salary
    st.subheader('Box Plot: EstimatedSalary')
    fig, ax = plt.subplots()
    sns.boxplot(x='Purchased', y='EstimatedSalary', data=df, hue='Purchased', palette={0: "blue", 1: "green"}, ax=ax)
    st.pyplot(fig)
    st.write('<span style="font-size:24px;">0-Not Purchased</span> <span style="font-size:24px;">1-Purchased</span>', unsafe_allow_html=True)

    # Scatter plot between 'EstimatedSalary' and 'Purchased' with light blue background
    st.subheader('Scatter Plot: EstimatedSalary vs Purchased')
    fig, ax = plt.subplots()
    sns.scatterplot(x='EstimatedSalary', y='Purchased', data=df, hue='Purchased', palette={0: "red", 1: "green"}, ax=ax)
    plt.xlabel('Estimated Salary')
    plt.ylabel('Purchased')
    st.pyplot(fig)  
    st.write('<span style="font-size:24px;">0-Not Purchased</span> <span style="font-size:24px;">1-Purchased</span>', unsafe_allow_html=True)


    return clf

# Main function
def main():
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        # Train the model
        clf = train_model(df)
        
    else:
        st.info('Awaiting for CSV file to be uploaded.')

if __name__ == '__main__':
    main()