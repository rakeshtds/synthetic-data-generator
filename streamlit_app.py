import streamlit as st
import os
import pandas as pd
import numpy as np
from faker import Faker
import json
import datetime
import random
from typing import List, Dict, Any
from anthropic import Anthropic
import ast

# Initialize Anthropic client
anthropic = Anthropic()

class SchemaParser:
    def __init__(self):
        self.anthropic = Anthropic()
    
    def parse_natural_language(self, description: str) -> List[Dict]:
        """Parse natural language description into schema definition"""
        prompt = f"""Given the following description of a data schema, convert it into a structured format.
        Description: {description}
        
        Return a Python list of dictionaries where each dictionary represents a field with:
        - name: field name (snake_case)
        - type: data type (string, integer, float, date, email, phone, name, address, company)
        - constraints: dictionary of constraints (if any)
        
        Example output format:
        [
            {{"name": "full_name", "type": "name", "constraints": {{}}}},
            {{"name": "age", "type": "integer", "constraints": {{"min": 18, "max": 65}}}}
        ]
        
        Provide only the Python list, no other text."""

        response = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            # Extract and parse the schema from Claude's response
            schema = ast.literal_eval(response.content[0].text)
            return schema
        except:
            st.error("Failed to parse schema. Please try rephrasing your description.")
            return []

class DataGenerator:
    def __init__(self):
        self.faker = Faker()
        self.anthropic = Anthropic()
        
    def generate_with_rules(self, schema: List[Dict], rules: str, num_records: int) -> pd.DataFrame:
        """Generate data based on schema and additional rules"""
        # First generate basic data
        data = []
        for _ in range(num_records):
            record = {}
            for field in schema:
                record[field["name"]] = self.generate_value(
                    field["type"],
                    field["constraints"]
                )
            data.append(record)
        
        # Apply additional rules if provided
        if rules:
            prompt = f"""Given this dataset and rules, modify the data to comply with the rules.
            Dataset (first 3 records for reference):
            {json.dumps(data[:3], indent=2)}
            
            Rules:
            {rules}
            
            Provide Python code that will transform the data to meet these rules.
            Return only the transformation code, no explanations."""

            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                # Execute the transformation code
                transformation_code = response.content[0].text
                local_dict = {"data": data, "pd": pd, "np": np}
                exec(transformation_code, globals(), local_dict)
                data = local_dict["data"]
            except Exception as e:
                st.warning(f"Failed to apply some rules: {str(e)}")
        
        return pd.DataFrame(data)

    def generate_value(self, field_type: str, constraints: Dict[str, Any]) -> Any:
        if field_type == "name":
            return self.faker.name()
        elif field_type == "email":
            return self.faker.email()
        elif field_type == "phone":
            return self.faker.phone_number()
        elif field_type == "integer":
            min_val = constraints.get('min', 0)
            max_val = constraints.get('max', 100)
            return random.randint(min_val, max_val)
        elif field_type == "float":
            min_val = constraints.get('min', 0.0)
            max_val = constraints.get('max', 100.0)
            return round(random.uniform(min_val, max_val), 2)
        elif field_type == "date":
            start_date = constraints.get('start_date', datetime.date(2000, 1, 1))
            end_date = constraints.get('end_date', datetime.date(2023, 12, 31))
            return self.faker.date_between(start_date=start_date, end_date=end_date)
        elif field_type == "address":
            return self.faker.address()
        elif field_type == "company":
            return self.faker.company()
        return None

def main():
    st.title("GenAI Synthetic Data Generator")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
        num_records = st.number_input("Number of records to generate", 
                                    min_value=1, 
                                    max_value=10000, 
                                    value=100)
    
    # Main interface
    st.header("Schema Definition")
    schema_description = st.text_area(
        "Describe your data schema in natural language",
        help="Example: Create a table with full name, age between 18-65, email, and company name"
    )
    
    # Additional rules
    st.header("Data Generation Rules")
    rules = st.text_area(
        "Specify additional rules for data generation (optional)",
        help="Example: Age should follow normal distribution with mean 35. Email domains should be mostly gmail.com"
    )
    
    if schema_description and st.button("Generate Data"):
        if not api_key:
            st.error("Please enter your Anthropic API Key in the sidebar")
            return
            
        with st.spinner("Parsing schema..."):
            parser = SchemaParser()
            schema = parser.parse_natural_language(schema_description)
            
            if schema:
                st.subheader("Parsed Schema")
                st.json(schema)
                
                with st.spinner("Generating data..."):
                    generator = DataGenerator()
                    df = generator.generate_with_rules(schema, rules, num_records)
                    
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
