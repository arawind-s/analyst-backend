import os
import json
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

class DataAnalysisAgent:
    def __init__(self):
        # Initialize the new Google GenAI client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.df = None
    
    def load_data(self, file_path: str):
        """Load CSV or Excel file into pandas DataFrame"""
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Handle NaN values - replace with 0 for numeric columns
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)
        
        # Replace NaN with empty string for object columns
        object_columns = self.df.select_dtypes(include=['object']).columns
        self.df[object_columns] = self.df[object_columns].fillna("")
        
        # Get preview with cleaned data
        preview_df = self.df.head(3)
        
        return {
            "columns": self.df.columns.tolist(),
            "row_count": len(self.df),
            "preview": preview_df.to_dict('records')
        }
    
    def analyze_query(self, query: str):
        """Analyze user query and generate chart configuration"""
        
        if self.df is None:
            return {"error": "No data loaded"}
        
        # Get data info
        data_summary = {
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "sample_data": self.df.head(5).to_dict('records'),
            "row_count": len(self.df)
        }
        
        prompt = f"""
You are a data analysis agent. Based on the user's query and dataset, generate a chart configuration.

Dataset Information:
{json.dumps(data_summary, indent=2)}

User Query: {query}

Tasks:
1. Analyze what the user wants to visualize
2. Determine the best chart type (bar, line, pie, scatter, area, doughnut)
3. Process the data (aggregate, filter, group as needed)
4. Generate the chart configuration

Return JSON in this exact format:
{{
  "chartType": "bar|line|pie|scatter|area|doughnut",
  "data": {{
    "labels": ["label1", "label2", ...],
    "datasets": [
      {{
        "label": "Dataset Name",
        "data": [value1, value2, ...],
        "backgroundColor": ["#color1", "#color2", ...] (optional)
      }}
    ]
  }},
  "config": {{
    "title": "Chart Title",
    "xAxisLabel": "X Axis Label",
    "yAxisLabel": "Y Axis Label"
  }},
  "interpretation": "Brief explanation of what is being shown",
  "processingSteps": ["step1", "step2", ...]
}}

Important:
- Actually process the data based on the query (aggregate, sum, count, filter, etc.)
- Return real processed values, not placeholders
- Choose appropriate chart type for the data
- Keep labels concise
"""

        try:
            # Use the new SDK's generate_content method
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'response_mime_type': 'application/json'
                }
            )
            
            result = json.loads(response.text)
            
            # Execute the actual data processing based on interpretation
            processed_result = self._execute_data_processing(query, result)
            
            return processed_result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _execute_data_processing(self, query: str, ai_result: dict):
        """Execute actual data processing using pandas"""
        
        try:
            chart_type = ai_result.get("chartType", "bar")
            query_lower = query.lower()
            
            # Simple heuristics for common queries
            if any(word in query_lower for word in ["total", "sum", "sales by", "revenue by"]):
                # Group by first categorical column and sum numeric columns
                cat_cols = self.df.select_dtypes(include=['object']).columns
                num_cols = self.df.select_dtypes(include=['number']).columns
                
                if len(cat_cols) > 0 and len(num_cols) > 0:
                    grouped = self.df.groupby(cat_cols[0])[num_cols[0]].sum()
                    
                    ai_result["data"]["labels"] = grouped.index.tolist()
                    ai_result["data"]["datasets"][0]["data"] = grouped.values.tolist()
            
            elif any(word in query_lower for word in ["count", "number of", "how many"]):
                # Count by category
                cat_cols = self.df.select_dtypes(include=['object']).columns
                
                if len(cat_cols) > 0:
                    counts = self.df[cat_cols[0]].value_counts()
                    
                    ai_result["data"]["labels"] = counts.index.tolist()
                    ai_result["data"]["datasets"][0]["data"] = counts.values.tolist()
            
            elif any(word in query_lower for word in ["average", "mean", "avg"]):
                # Average by category
                cat_cols = self.df.select_dtypes(include=['object']).columns
                num_cols = self.df.select_dtypes(include=['number']).columns
                
                if len(cat_cols) > 0 and len(num_cols) > 0:
                    grouped = self.df.groupby(cat_cols[0])[num_cols[0]].mean()
                    
                    ai_result["data"]["labels"] = grouped.index.tolist()
                    ai_result["data"]["datasets"][0]["data"] = [round(x, 2) for x in grouped.values.tolist()]
            
            return ai_result
            
        except Exception as e:
            # If processing fails, return AI result as is
            ai_result["processingNote"] = f"Used AI interpretation: {str(e)}"
            return ai_result