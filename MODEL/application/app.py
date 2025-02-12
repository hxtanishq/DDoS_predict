import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import os
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.grid_options_builder import GridOptionsBuilder
import plotly.graph_objects as go

# Home Page
def show_introduction():
    st.title("Predicting DDoS Attacks with LSTM Time-Series Model")
    
    st.subheader("Guide: Ms. Arti Singh")
    st.subheader("Co-Guide: Priyanka Abhale")

    st.markdown("""
    **Submitted By:**
    - **Tanishq Gupta (BEAD21168)**
    - **Sejal Kangane (BEAD21125)**
    - **Vaibhav Muley (BEAD21126)**
    - **Kushal Khachane (BEAD21127)**
    """)

    st.write("## Problem Statement")
    st.write("Developing an LSTM-based time-series model to enhance DDoS attack prediction accuracy while minimizing false positives and adapting to evolving attack patterns.")

    st.write("## Abstract")
    st.write("""
    Due to the rise in frequency and efficiency of Distributed Denial of Service (DDoS) attacks, suitable and timely security measures are required.
    This project presents a web application that employs Long Short-Term Memory (LSTM) deep learning for identifying DDoS assaults.
    The system is designed for high-speed, low-latency detection of DDoS events by analyzing network connection patterns.
    """)
    
    st.write("## Motivation")
    st.markdown("""
    - Improve DDoS Attack Prediction Accuracy
    - Enhance Network Security
    - Address limitations of existing models
    - Enable timely attack mitigation
    - Protect Online Systems
    """)

# Project Details Page

def show_project_details():
    st.title("Project Details ")

    st.write("## System Architecture")
    selected_image = f"ddos_architecture.png"
    dataset_dir = r"C:\Users\tanis\my_folder\GEN_AI\ddos_code\application"
    if selected_image:
        # Open the selected image 
        image_path = os.path.join(dataset_dir, selected_image)
        image = Image.open(image_path)
        # Display the image
        st.image(image, caption=f'Selected Image: {selected_image}', use_column_width=True)
        st.write("Image loaded successfully!")


    st.write("## Problem Description")
    st.write("""
    DDoS attacks are increasingly becoming a major issue regarding the stability and availability of online systems. This project seeks to develop
    a predictive DDoS system using an LSTM time-series model that enhances detection speed and accuracy.
    """)

    st.write("## Research Questions")
    st.markdown("""
    1. How can LSTM architectures be optimized to accurately detect DDoS attacks amidst noisy network traffic?
    2. What is the impact of different feature engineering techniques on the performance of DDoS attack prediction models?
    3. How can the proposed LSTM model be integrated into existing network security infrastructure for real-time protection?
    4. What is the optimal window size for capturing relevant historical data for accurate DDoS attack prediction?
    5. How does the performance of the LSTM model compare to other machine learning techniques in detecting DDoS attacks?
    """)

    st.write("## Objectives")
    st.markdown("""
    - Develop an LSTM-based Model for Accurate DDoS Attack Prediction
    - Evaluate the Model's Performance using Relevant Metrics
    - Identify Potential Improvements
    - Analyze Network Traffic Patterns to Enhance Detection Capabilities
    - Create a Web-Based Application for Real-time DDoS Attack Prediction
    """)

    st.write("## Potential Outcomes")
    st.markdown("""
    1. Improved DDoS Attack Prediction Accuracy
    2. Enhanced Network Security through early Detection and Mitigation
    3. Increased System Resilience
    4. Reduced Financial Losses associated with DDoS Downtime
    5. Contribution to Cybersecurity Research
    """)

    st.write("## Applications")
    st.markdown("""
    1. Protect Critical Systems
    2. Improve Online Service Availability
    3. Support Incident Response
    4. Contribution to Cybersecurity Research
    5. Integration into Cybersecurity Toolkit
    """)

 
@st.cache_data
def load_data(): 
    file = f"data.csv"
    ddos_data=pd.read_csv(file)
    return ddos_data
 
def visualize_label_distribution(df):
    st.subheader("Label Distribution")
    label_counts = df[' Label'].value_counts()
    
    top_labels = label_counts.nlargest(5)
    other_count = label_counts.iloc[5:].sum()
    if other_count > 0:
        top_labels['Other'] = other_count
        
    fig, ax = plt.subplots()
    ax.pie(top_labels, labels=top_labels.index, autopct='%1.1f%%', startangle=85)
    ax.axis('equal')
    st.pyplot(fig)
 

df = load_data()
connection_by_time_b=df[df[' Label']=='BENIGN'].groupby(' Timestamp').size()
connection_by_time_nb=df[df[' Label']!='BENIGN'].groupby(' Timestamp').size()
plt.figure(figsize=(12, 6))
plt.plot(connection_by_time_nb.index, connection_by_time_nb.values, marker='o', linestyle='-',label='DDoS attack')
plt.plot(connection_by_time_b.index, connection_by_time_b.values, marker='o', linestyle='-', label='Benign')
plt.title('Benign/DDoS Connections Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
    #plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
# st.markdown("""
#                 Although this dataset contains more connections associated with DDoS attacks, 
#                 the connections labeled as benign are more evenly distributed over time. 
#                 Conversely, connections labeled as DDoS attacks are concentrated within a short time interval, 
#                 with a high number of requests. Additionally, within this time interval, 
#                 the number of connections labeled as benign is also relatively high.
#                 DDoS attacks are often characterized by a large number of requests sent simultaneously or over a very short period of time.")
#                 """)


def visualize_top_features_and_distributions(df):
    # Create two columns for side by side display
    col1, col2 = st.columns(2)     
    with col1:
        st.subheader("Top 20 Important Features")
        img_file = r"C:\Users\tanis\my_folder\GEN_AI\ddos_code\application\top20.png"
        st.image(img_file)
        
        
    with col2:
        # Label Distribution
        st.subheader("Label Distribution")
        label_counts = df[' Label'].value_counts()
        fig_labels = plt.figure(figsize=(8, 8))
        plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of DDoS Attack Types')
        plt.axis('equal')
        st.pyplot(fig_labels)
        plt.close()


def visualize_labels_distribution(df):
    
    label_counts=df[' Label'].value_counts()
    plt.figure(figsize=(3, 3))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of DDoS Attack Types')
    plt.axis('equal') 
    plt.show()
    
    
def visualize_packet_distribution(df):
    st.subheader("Packets Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Flow Bytes/s'], df[' Total Fwd Packets'], label='Total Fwd Packets', color='blue')
    # ax.plot(df[' Timestamp'], df[' Total Backward Packets'], label='Total Backward Packets', color='red')
    plt.title('Packets over Time')
    # plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.legend()
    st.pyplot(fig)
 
def show_implementation(df):
    st.title("Project Implementation")

    st.write("## Data Preprocessing")
    st.write("Various preprocessing steps such as normalization, label encoding, and feature selection were performed.")
    visualize_labels_distribution(df)
    st.write("## Feature Engineering")
    st.write("The following features were used in the prediction model:")
    st.dataframe(df.columns[1:])  # Show feature columns
    
    st.write("## Visualizations")
    visualize_top_features_and_distributions(df)

    # visualize_traffic_over_time(df)
    visualize_packet_distribution(df)
 












def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Project Details", "Implementation","CopyRight"])
    ddos_data = load_data()
    ddos_data[' Timestamp'] = pd.to_datetime(ddos_data[' Timestamp'])
    ddos_data.sort_values(by=' Timestamp', inplace=True)
    constant_columns=ddos_data.columns[ddos_data.nunique()==1]
    ddos_data.drop(['Flow ID',' Fwd Header Length.1'],axis=1,inplace=True)
    ddos_data.drop(columns=constant_columns,inplace=True)
    
    
    
    if selection == "Home":
        show_introduction()   
    elif selection == "Project Details":
        show_project_details()
    elif selection == "Implementation":
        show_implementation(ddos_data)
        load_model_comparison()
    elif selection == "CopyRight":
        pass

        
        
        
        
        
        
        
        
def load_model_comparison():
    st.title("Model Performance Comparison")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload model comparison CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file, index_col=0)
        
        # Display the results table
        st.subheader("Model Comparison Table")
        st.dataframe(df.style.format("{:.4f}"))
        
        # Create visualizations
        create_comparison_visualizations(df)
        
        

def load_model_comparison():
    st.title("Model Performance Comparison")
    file_pat = r"C:\Users\tanis\my_folder\GEN_AI\ddos_code\application\model_compare.csv"
    # Read CSV file directly from system
    # Replace 'model_comparison.csv' with your actual file path
    try:
        df = pd.read_csv( file_pat, index_col=0)
        
        # Display the results table
        st.subheader("Model Comparison Table")
        st.dataframe(df.style.format("{:.4f}"))
        
        # Create visualizations
        create_comparison_visualizations(df)
        
        
    except FileNotFoundError:
        st.error("Error: CSV file not found. Please ensure 'model_comparison.csv' exists in the same directory as the script.")

def create_comparison_visualizations(df):
    st.subheader("Performance Visualization")
    
    # Metrics comparison plot
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    for model in df.index:
        fig.add_trace(go.Scatter(
            x=metrics_to_plot,
            y=df.loc[model, metrics_to_plot],
            name=model,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Model Performance Metrics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=600
    )
    st.plotly_chart(fig)
    
    # Training time comparison
    if 'Training Time (s)' in df.columns:
        fig_time = go.Figure(data=[
            go.Bar(
                x=df.index,
                y=df['Training Time (s)'],
                name='Training Time'
            )
        ])
        
        fig_time.update_layout(
            title="Training Time Comparison",
            xaxis_title="Models",
            yaxis_title="Time (seconds)",
            height=400
        )
        st.plotly_chart(fig_time)
    


if __name__ == "__main__":
    main()
