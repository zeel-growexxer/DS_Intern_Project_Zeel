import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("InfoQuest")

def send_query(question):
    url = 'http://127.0.0.1:5000/query'
    payload = {'question': question}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def generate_visualization(df, question):
    visualized = False  # Flag to track if a visualization has been generated

    for col in df.columns:
        if not visualized:  # Only generate one visualization per question
            if df[col].dtype == 'object':
                # Check if the column is categorical (limited unique values)
                if df[col].nunique() <= 10:
                    fig = px.bar(df, x=col, title=f"Bar plot of {col}")
                    fig.update_layout(xaxis_title=col, yaxis_title="Count", showlegend=True)
                    st.plotly_chart(fig)
                    visualized = True
                else:
                    st.write(f"No appropriate visualization found for column {col}.")
            elif df[col].dtype in ['int64', 'float64']:
                # Check the nature of numerical columns for appropriate plots
                if df[col].nunique() > 10:
                    fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=20)
                    fig.update_layout(xaxis_title=col, yaxis_title="Count", showlegend=True)
                    st.plotly_chart(fig)
                    visualized = True
                elif df[col].nunique() > 1:
                    fig = px.box(df, y=col, title=f"Box plot of {col}")
                    fig.update_layout(xaxis_title=col, yaxis_title="Value", showlegend=True)
                    st.plotly_chart(fig)
                    visualized = True
                else:
                    st.write(f"No appropriate visualization found for column {col}.")
        
        if not visualized:  # Check for scatter plot if not yet visualized
            if len(df[col].dropna().unique()) <= 10 and df[col].dtype in ['int64', 'float64']:
                for col2 in df.columns:
                    if col2 != col and df[col2].dtype in ['int64', 'float64'] and len(df[col2].dropna().unique()) <= 10:
                        fig = px.scatter(df, x=col, y=col2, title=f"Scatter plot of {col} vs {col2}")
                        fig.update_layout(xaxis_title=col, yaxis_title=col2, showlegend=True)
                        st.plotly_chart(fig)
                        visualized = True
                        break  # Exit loop after first valid scatter plot

        if not visualized:  # Check for pie chart if not yet visualized
            if df[col].dtype == 'object' and df[col].nunique() > 1:
                fig = px.pie(df, names=col, title=f"Pie chart of {col}")
                st.plotly_chart(fig)
                visualized = True

    if not visualized:
        st.write("No appropriate visualization found for this question.")

question = st.text_input("Type your question here:")

if st.button("Send"):
    if question:
        response = send_query(question)
        if isinstance(response, dict):
            results = response.get('results')
            if results:
                df = pd.DataFrame(results)
                generate_visualization(df, question)
            st.write(f"Response: {response.get('response')}")
        else:
            st.write(response)
