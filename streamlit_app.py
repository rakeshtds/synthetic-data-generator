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
        prompt = '''You are a data schema parser. Convert this natural language description into a structured schema format.
        
Description: {description}

Rules:
1. Convert each described field into a dictionary with: name, type, and constraints
2. Use snake_case for field names
3. Only use these types: name, email, phone, integer, float, date, address, company
4. Extract any mentioned constraints (min/max for numbers, date ranges, etc.)
5. If no constraints are mentioned, use empty dict {{}}

Example 1:
Input: "Create a table with full name, age between 18-65, and email"
Output: [
    {{"name": "full_name", "type": "name", "constraints": {{}}}},
    {{"name": "age", "type": "integer", "constraints": {{"min": 18, "max": 65}}}},
    {{"name": "email", "type": "email", "constraints": {{}}}}
]

Example 2:
Input: "I need employee data with name, company, and joining date between 2020 and 2024"
Output: [
    {{"name": "employee_name", "type": "name", "constraints": {{}}}},
    {{"name": "company", "type": "company", "constraints": {{}}}},
    {{"name": "joining_date", "type": "date", "constraints": {{"start_date": "2020-01-01", "end_date": "2024-12-31"}}}}
]

Return ONLY a valid Python list of dictionaries that can be parsed with ast.literal_eval(). No other text.'''.format(description=description)

        response = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
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
            prompt = '''Given this dataset and rules, modify the data to comply with the rules.
Dataset (first 3 records for reference):
{data}

Rules:
{rules}

Provide Python code that will transform the data to meet these rules.
Return only the transformation code, no explanations.'''.format(
                data=json.dumps(data[:3], indent=2),
                rules=rules
            )

            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
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
    st.set_page_config(page_title="FIC DATA Generator", layout="wide")
    
    # Custom CSS for title styling
    st.markdown('''
<style>
.title {
    text-align: center;
    color: #2E4053;
    padding: 20px;
    border-radius: 6px;
    margin-bottom: 32px;
    background-color: #f8f9fa;
}
</style>
    ''', unsafe_allow_html=True)
    
    st.markdown("<h1 class='title'>FIC DATA Generator</h1>", unsafe_allow_html=True)
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
        num_records = st.number_input(
            "Number of records to generate", 
            min_value=1, 
            max_value=10000, 
            value=100
        )
    
    # Main interface - Tabs for different schema definition methods
    schema_tab1, schema_tab2 = st.tabs(["Natural Language Input", "Manual Schema Builder"])
    
    schema = None
    
    with schema_tab1:
        st.markdown('''
### Natural Language Schema Definition
Describe your data schema in plain English. You can specify:
- Field names and types
- Constraints (e.g., age ranges, date ranges)
- Any special requirements

**Example inputs:**
1. "Create a table with full name, age between 18-65, and email"
2. "I need employee data with name, company, and joining date between 2020 and 2024"
3. "Generate customer data with name, phone number, address, and age above 21"
        ''')
        
        schema_description = st.text_area(
            "Enter your schema description",
            placeholder="Example: Create a table with full name, age between 18-65, email, and company name",
            help="Be specific about any constraints like age ranges or date ranges"
        )
        
        if schema_description and st.button("Parse Schema", key="parse_nl"):
            with st.spinner("Parsing schema description..."):
                parser = SchemaParser()
                try:
                    schema = parser.parse_natural_language(schema_description)
                    if schema:
                        st.success("Schema parsed successfully!")
                        st.subheader("Parsed Schema")
                        st.json(schema)
                    else:
                        st.error("Failed to parse schema. Please try a different description.")
                except Exception as e:
                    st.error(f"Error parsing schema: {str(e)}")
                    st.info("Try a simple example like: 'Create a table with full name, age between 18-65, and email'")
            
    with schema_tab2:
        # Available field types
        FIELD_TYPES = [
            "name",
            "email",
            "phone",
            "integer",
            "float",
            "date",
            "address",
            "company"
        ]
        
        # Initialize manual schema in session state if not exists
        if 'manual_schema' not in st.session_state:
            st.session_state.manual_schema = []
            
        # Add new field form
        with st.expander("Add New Field", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_field_name = st.text_input("Field Name")
            with col2:
                new_field_type = st.selectbox("Field Type", FIELD_TYPES)
                
            # Additional constraints based on field type
            constraints = {}
            if new_field_type in ["integer", "float"]:
                min_val = st.number_input("Minimum Value", value=0)
                max_val = st.number_input("Maximum Value", value=100)
                constraints = {"min": min_val, "max": max_val}
            elif new_field_type == "date":
                start_date = st.date_input("Start Date", datetime.date(2000, 1, 1))
                end_date = st.date_input("End Date", datetime.date(2023, 12, 31))
                constraints = {"start_date": start_date, "end_date": end_date}
                
            if st.button("Add Field", key="add_field"):
                if new_field_name and new_field_type:
                    st.session_state.manual_schema.append({
                        "name": new_field_name,
                        "type": new_field_type,
                        "constraints": constraints
                    })
        
        # Display current schema
        with st.expander("Current Schema", expanded=True):
            if st.session_state.manual_schema:
                schema_df = pd.DataFrame(st.session_state.manual_schema)
                st.dataframe(schema_df)
                
                if st.button("Clear Schema"):
                    st.session_state.manual_schema = []
                    st.rerun()
                    
                schema = st.session_state.manual_schema
            else:
                st.info("No fields added yet. Use the form above to add fields to your schema.")
    
    # Additional rules
    st.header("Data Generation Rules")
    rules = st.text_area(
        "Specify additional rules for data generation (optional)",
        help="Example: Age should follow normal distribution with mean 35. Email domains should be mostly gmail.com"
    )
    
    # Generate Data button outside tabs
    if schema and st.button("Generate Data", key="generate"):
        if not api_key:
            st.error("Please enter your Anthropic API Key in the sidebar")
            return
            
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
