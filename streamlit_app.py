import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import json
import datetime
import random
from typing import List, Dict, Any

# Initialize Faker
fake = Faker()

class DataGenerator:
    def __init__(self):
        self.faker = Faker()
        
    def generate_value(self, field_type: str, constraints: Dict[str, Any]) -> Any:
        if field_type == "Full Name":
            return self.faker.name()
        elif field_type == "Email":
            return self.faker.email()
        elif field_type == "Phone":
            return self.faker.phone_number()
        elif field_type == "Integer":
            min_val = constraints.get('min', 0)
            max_val = constraints.get('max', 100)
            return random.randint(min_val, max_val)
        elif field_type == "Float":
            min_val = constraints.get('min', 0.0)
            max_val = constraints.get('max', 100.0)
            return round(random.uniform(min_val, max_val), 2)
        elif field_type == "Date":
            start_date = constraints.get('start_date', datetime.date(2000, 1, 1))
            end_date = constraints.get('end_date', datetime.date(2023, 12, 31))
            return self.faker.date_between(start_date=start_date, end_date=end_date)
        elif field_type == "Address":
            return self.faker.address()
        elif field_type == "Company":
            return self.faker.company()
        return None

def main():
    st.title("Synthetic Data Generator")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Number of records
    num_records = st.sidebar.number_input("Number of records to generate", 
                                        min_value=1, 
                                        max_value=10000, 
                                        value=100)
    
    # Available field types
    FIELD_TYPES = [
        "Full Name",
        "Email",
        "Phone",
        "Integer",
        "Float",
        "Date",
        "Address",
        "Company"
    ]
    
    # Schema Builder
    st.header("Schema Builder")
    
    # Initialize schema in session state if not exists
    if 'schema' not in st.session_state:
        st.session_state.schema = []
    
    # Add new field form
    with st.expander("Add New Field", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            new_field_name = st.text_input("Field Name")
        with col2:
            new_field_type = st.selectbox("Field Type", FIELD_TYPES)
            
        # Additional constraints based on field type
        constraints = {}
        if new_field_type in ["Integer", "Float"]:
            min_val = st.number_input("Minimum Value", value=0)
            max_val = st.number_input("Maximum Value", value=100)
            constraints = {"min": min_val, "max": max_val}
        elif new_field_type == "Date":
            start_date = st.date_input("Start Date", datetime.date(2000, 1, 1))
            end_date = st.date_input("End Date", datetime.date(2023, 12, 31))
            constraints = {"start_date": start_date, "end_date": end_date}
            
        if st.button("Add Field"):
            if new_field_name and new_field_type:
                st.session_state.schema.append({
                    "name": new_field_name,
                    "type": new_field_type,
                    "constraints": constraints
                })
    
    # Display current schema
    st.subheader("Current Schema")
    if st.session_state.schema:
        schema_df = pd.DataFrame(st.session_state.schema)
        st.dataframe(schema_df)
        
        if st.button("Clear Schema"):
            st.session_state.schema = []
            st.experimental_rerun()
    else:
        st.info("No fields added yet. Use the form above to add fields to your schema.")
    
    # Generate Data
    if st.session_state.schema and st.button("Generate Data"):
        generator = DataGenerator()
        data = []
        
        with st.spinner("Generating synthetic data..."):
            for _ in range(num_records):
                record = {}
                for field in st.session_state.schema:
                    record[field["name"]] = generator.generate_value(
                        field["type"],
                        field["constraints"]
                    )
                data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Display preview
        st.subheader("Generated Data Preview")
        st.dataframe(df.head())
        
        # Download options
        st.subheader("Download Options")
        col1, col2 = st.columns(2)
        
        # CSV download
        csv = df.to_csv(index=False)
        col1.download_button(
            label="Download CSV",
            data=csv,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )
        
        # JSON download
        json_str = df.to_json(orient="records")
        col2.download_button(
            label="Download JSON",
            data=json_str,
            file_name="synthetic_data.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
