import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_classification.structured.bank_statement import extract_data_from_image, save_to_csv, store_csv_to_postgresql, create_visualizationss
from data_classification.structured.invoice import extract_from_description_to_csv, store_csv_to_postgresql as store_invoice_to_postgresql, create_visualizations as create_invoice_visualizations
from data_classification.structured.payslip import process_image_to_csv, store_csv_to_postgresql as store_payslip_to_postgresql, visualize_data as visualize_payslip_data
from data_classification.structured.profit_loss import extract_profit_loss_data, store_profit_loss_to_postgresql, create_visualizations
from data_classification.structured.pdf_processing import convert_pdf_to_csv, save_csv, store_data_in_db, generate_debit_credit_visualizations
from data_classification.semi_structured.api_json import process_json_data, create_json_visualizations
from data_classification.unstructured.unstructure import process_data
import pytesseract
import os
from sqlalchemy import create_engine

# Set the Tesseract path explicitly (for Linux or cloud-based environments)
if os.name == 'posix':  # Linux-based systems (e.g., Streamlit Cloud)
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update this path accordingly
elif os.name == 'nt':  # For Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Tesseract-OCR\\tesseract.exe'

if not os.path.exists('data'):
    os.makedirs('data')
st.title("Welcome to Bank Data Processing App! 🏦")
st.title("Bank Data Processing and Analysis")
st.markdown("""
- 💼 **Structured:** For OCR extracted from Images and PDFs
- 🔀 **Semi-Structured:** For JSON files
- 📂 **Unstructured:** For unlabelled text

Start by uploading your file to analyze!
""")

st.write("Upload the data of any format you want to analyse, understand and get clear perspective from them!")

data_type = st.selectbox("Select Data Type", ["Structured", "Semi-Structured", "Unstructured"])

if data_type:
    if data_type == "Structured":
        data_choice = st.radio("Choose Data Type", ("Bank Statement", "Invoice", "Payslip", "Profit and Loss", "Pdf processing"))

        if data_choice == "Bank Statement":
            st.subheader("Structured Data: Bank Statement OCR Upload")
            uploaded_file = st.file_uploader("Upload a Bank Statement Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    image_path = "uploaded_image.png"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    extracted_data = extract_data_from_image(image_path)

                if extracted_data:
                    csv_output_path = "data/transactions.csv"
                    save_to_csv(extracted_data, csv_output_path)
                    st.success(f"Extracted data saved to {csv_output_path}")
                    st.write("### Extracted Transactions")
                    st.dataframe(extracted_data)
                    st.download_button(
                        label="Download CSV",
                        data=open(csv_output_path, "rb").read(),
                        file_name="transactions.csv",
                        mime="text/csv"
                    )
                    db_config = {
                        "db_name": "bankstatement_db",
                        "user": "postgres",
                        "password": "newpassword",
                        "host": "localhost",
                        "port": "5432",
                        "table_name": "bankstatement_table"
                    }
                    with st.spinner("Storing data in the database..."):
                        store_csv_to_postgresql(csv_output_path, **db_config)
                    st.success(f"Data successfully stored in the PostgreSQL database ({db_config['table_name']})")

                    df = pd.read_csv(csv_output_path)
                    bar_chart_path = "data/bar_chart.png"
                    pie_chart_path = "data/pie_chart.png"

                    with st.spinner("Creating visualizations..."):
                        create_visualizationss(df, bar_chart_path, pie_chart_path)

                    st.write("### Visualizations")
                    st.image(bar_chart_path, caption="Bar Chart")
                    st.image(pie_chart_path, caption="Pie Chart")

                else:
                    st.error("No data was extracted from the image. Please upload a valid bank statement.")

        elif data_choice == "Invoice":
            st.subheader("Structured Data: Invoice OCR Upload")
            uploaded_file = st.file_uploader("Upload an Invoice Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    image_path = "uploaded_invoice.png"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    csv_output_path = "data/invoice_data.csv"
                    df = extract_from_description_to_csv(image_path, csv_output_path)

                st.success(f"Extracted invoice data saved to {csv_output_path}")
                st.write("### Extracted Invoice Data")
                st.dataframe(df)

                st.download_button(
                    label="Download CSV",
                    data=open(csv_output_path, "rb").read(),
                    file_name="invoice_data.csv",
                    mime="text/csv"
                )

                db_config = {
                    "db_name": "invoice_db",
                    "user": "postgres",
                    "password": "newpassword",
                    "host": "localhost",
                    "port": "5432",
                    "table_name": "invoice_table"
                }
                with st.spinner("Storing data in the database..."):
                    store_invoice_to_postgresql(csv_output_path, **db_config)

                st.success(f"Invoice data successfully stored in the PostgreSQL database ({db_config['table_name']})")
                visualization_paths = {
                    "bar_chart_path": "data/invoice_bar_chart.png",
                    "pie_chart_path": "data/invoice_pie_chart.png"
                }
                with st.spinner("Creating visualizations..."):
                    create_invoice_visualizations(df, visualization_paths["bar_chart_path"], visualization_paths["pie_chart_path"])
                st.write("### Visualizations")
                st.image(visualization_paths["bar_chart_path"], caption="Bar Chart")
                st.image(visualization_paths["pie_chart_path"], caption="Pie Chart")

        elif data_choice == "Payslip":
            st.subheader("Structured Data: Payslip OCR Upload")
            uploaded_file = st.file_uploader("Upload a Payslip Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    image_path = "uploaded_payslip.png"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    csv_output_path = "data/payslip_data.csv"
                    process_image_to_csv(uploaded_file, csv_output_path)

                st.success(f"Extracted payslip data saved to {csv_output_path}")
                df = pd.read_csv(csv_output_path)
                st.write("### Extracted Payslip Data")
                st.dataframe(df)

                st.download_button(
                    label="Download CSV",
                    data=open(csv_output_path, "rb").read(),
                    file_name="payslip_data.csv",
                    mime="text/csv"
                )
                db_config = {
                    "db_name": "payslip_db",
                    "user": "postgres",
                    "password": "newpassword",
                    "host": "localhost",
                    "port": "5432",
                    "table_name": "payslip_table"
                }
                with st.spinner("Storing data in the database..."):
                    store_payslip_to_postgresql(csv_output_path, **db_config)

                st.success(f"Payslip data successfully stored in the PostgreSQL database ({db_config['table_name']})")

                st.write("### Visualizations")
                with st.spinner("Creating visualizations..."):
                    visualize_payslip_data(csv_output_path)
					
        elif data_choice == "Profit and Loss":
            st.subheader("Structured Data: Profit and Loss OCR Upload")
            uploaded_file = st.file_uploader("Upload a Profit and Loss Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    image_path = "uploaded_profit_loss.png"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Define the path for saving CSV
                    csv_output_path = "data/profit_loss_data.csv"
                    profit_loss_data = extract_profit_loss_data(image_path, csv_output_path)

                    st.success(f"Extracted profit and loss data saved to {csv_output_path}")
                
                    # Display extracted data as DataFrame in Streamlit
                    st.write("### Extracted Profit and Loss Data")
                    st.dataframe(profit_loss_data)

                    # Add a download button for CSV file
                    st.download_button(
                        label="Download CSV",
                        data=open(csv_output_path, "rb").read(),
                        file_name="profit_loss_data.csv",
                        mime="text/csv"
                    )

                    # Define database configuration
                    db_config = {
                        "db_name": "profitloss_db",
                        "user": "postgres",
                        "password": "newpassword",
                        "host": "localhost",
                        "port": "5432",
                        "table_name": "profitloss_table"
                    }

                    with st.spinner("Storing data in the database..."):
                        store_profit_loss_to_postgresql(profit_loss_data, **db_config)

                    st.success(f"Profit and Loss data successfully stored in the PostgreSQL database ({db_config['table_name']})")
                    
                    # Paths for visualizations
                    visualization_paths = {
                        "bar_chart_path": "data/profit_loss_bar_chart.png",
                        "pie_chart_path": "data/profit_loss_pie_chart.png"
                    }

                    # Create and display visualizations
                    with st.spinner("Creating visualizations..."):
                        create_visualizations(profit_loss_data, visualization_paths["bar_chart_path"], visualization_paths["pie_chart_path"])

                       



        elif data_choice == "Pdf processing":
                st.subheader("Structured Data: Pdf Processing Upload")
                uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

                if uploaded_file is not None:
                    pdf_path = "uploaded_file.pdf"
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    csv_output_path = "data/newtransactions.csv"
                    df = convert_pdf_to_csv(pdf_path, csv_output_path)

                    # Ensure the CSV is saved and data is extracted
                    save_csv(df, csv_output_path)

                    st.subheader("Extracted Data")
                    st.dataframe(df)
                    with open(csv_output_path, "rb") as f:
                        st.download_button(
                        label="Download CSV",
                        data=f,
                        file_name="newtransactions.csv",
                        mime="text/csv"
                        )

                    # Now that `df` is defined, we can call store_data_in_db safely
                    db_config = {
                    "db_name": "transactions_db",
                    "user": "postgres",
                    "password": "newpassword",  
                    "host": "localhost",
                    "port": "5432",
                     "table_name": "transactions_table"
                    }

                    store_data_in_db(df, **db_config)

                    # Generate bar chart for debit vs credit and show it
                    bar_chart_path = generate_debit_credit_visualizations(csv_output_path)
                    st.image(bar_chart_path, caption="Debit vs Credit Comparison", use_container_width=True)


    elif data_type == "Semi-Structured":
        st.subheader("Semi-Structured Data Upload (JSON-based)")
        uploaded_file = st.file_uploader("Upload a JSON File", type=["json"])

        if uploaded_file:
  
            json_path = "uploaded_data.json"
            with open(json_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            json_data = process_json_data(json_path)

            st.success("JSON data processed successfully")

            st.write("### Processed JSON Data")
            st.dataframe(json_data)

            csv_output_path = "data/semi_structured_data.csv"
            json_data.to_csv(csv_output_path, index=False)
            st.download_button(
                label="Download CSV",
                data=open(csv_output_path, "rb").read(),
                file_name="semi_structured_data.csv",
                mime="text/csv"
            )
            bar_chart_path = "data/semi_structured_bar_chart.png"
            pie_chart_path = "data/semi_structured_pie_chart.png"

            create_json_visualizations(json_data, bar_chart_path, pie_chart_path)
            st.write("### Visualizations")
            st.image(bar_chart_path, caption="Bar Chart")
            st.image(pie_chart_path, caption="Pie Chart")


    elif data_type == "Unstructured":
        st.subheader("Unstructured Data Upload (CSV-based)")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
        # Load the uploaded file into a DataFrame
            df = pd.read_csv(uploaded_file)

        # Number of Clusters input from user
            num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=3, value=3, step=1)

        # Process the data with the selected number of clusters (using the function from data_processing.py)
            processed_df = process_data(df, num_clusters)

        # Display processed data
            st.subheader("Processed Data")
            st.dataframe(processed_df)

        # Scatter Plot: Transactions with KMeans Clusters
            st.subheader("Scatter Plot of Transactions with KMeans Clusters")
            fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes
            sns.scatterplot(data=processed_df, x='Amount', y='Date', hue='Cluster_Label', palette='Set2', s=120, ax=ax)
            ax.set_title('Scatter Plot of Transactions with KMeans Clusters', fontsize=16)
            ax.set_xlabel('Amount', fontsize=14)
            ax.set_ylabel('Date', fontsize=14)
            st.pyplot(fig)  # Pass figure explicitly to st.pyplot()

            # Bar Chart: Transaction Count in Each Cluster
            st.subheader("Transaction Count in Each Cluster")
            fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes
            sns.countplot(data=processed_df, x='Cluster_Label', palette='Set2', edgecolor='black', linewidth=1.5, ax=ax)
            ax.set_title('Transaction Count in Each Cluster', fontsize=16)
            ax.set_xlabel('Cluster Label', fontsize=14)
            ax.set_ylabel('Count', fontsize=14)

        # Annotate bars with count values inside the bars
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2, height / 2,
                    f'{int(height)}', ha='center', va='center', fontsize=12, color='white')

                st.pyplot(fig)  # Pass figure explicitly to st.pyplot()

        # Pie Chart: Distribution of Transaction Categories (New vs Regular)
                st.subheader("Distribution of Transaction Categories (New vs Regular)")
                category_counts = processed_df['Transaction_Category'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
                ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'], startangle=90)
                ax.set_title('Distribution of Transaction Categories (New vs Regular)', fontsize=14)
                ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle
                st.pyplot(fig)  # Pass figure explicitly to st.pyplot()

        # Download Button for Processed Data
                st.subheader("Download Processed Data")
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name='processed_transactions.csv',
                mime='text/csv'
                )
