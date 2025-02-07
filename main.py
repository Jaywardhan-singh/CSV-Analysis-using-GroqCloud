import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GroqCloud API details
GROQCLOUD_API_KEY = os.getenv("GROQCLOUD_API_KEY")
GROQCLOUD_API_URL = os.getenv("GROQCLOUD_API_URL")

# Function to handle CSV file upload
def handle_csv_upload(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Function to get response from GroqCloud
def get_groqcloud_response(data, question):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQCLOUD_API_KEY}"
    }
    payload = {
        "model": "llama3-8b-8192",  # Replace with the specific GroqCloud model if required
        "messages": [{"role": "user", "content": f"Based on this data: {data}, {question}"}]
    }
    response = requests.post(GROQCLOUD_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI setup
st.title("CSV File Analysis with GroqCloud API")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    # Handle file upload
    df = handle_csv_upload(uploaded_file)
    st.write("First 5 Rows of the Dataframe:")
    st.write(df.head(5))

    # User inputs dynamic query
    question = st.text_area("Ask anything related to the uploaded data:")

    if st.button("Generate Response"):
        if question:
            with st.spinner("Generating response..."):
                data_preview = df.head(5).to_dict()  # Send a preview of the data
                response = get_groqcloud_response(data_preview, question)
                st.write("Response:")
                st.write(response)
        else:
            st.warning("Please enter a question.")
else:
    st.warning("Please upload a CSV file.")

# Option to Download Data as CSV
if st.button("Download Data as CSV"):
    df.to_csv("output_data.csv", index=False)
    st.download_button(label="Download CSV", data=open("output_data.csv", 'rb'), file_name="output_data.csv")
