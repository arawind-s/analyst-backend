import os
import json
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Dict, Any, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DataAnalysisAgent:
    """Production-grade data analysis agent using Gemini 2.5 Flash with Google Search grounding"""
    
    def __init__(self):
        try:
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = "gemini-2.5-flash"
            self.df = None
            self.column_info = None
            self.date_columns = []
            logger.info(f"Initialized {self.model} successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    def _auto_detect_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and parse date columns"""
        date_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().head(100)
                    if len(sample) == 0:
                        continue
                    
                    converted = pd.to_datetime(sample, errors='coerce')
                    success_rate = converted.notna().sum() / len(sample)
                    
                    if success_rate >= 0.8:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        date_columns.append(col)
                        logger.info(f"Auto-detected date column: {col} (success rate: {success_rate:.2%})")
                        
                except Exception as e:
                    logger.debug(f"Column {col} is not a date column: {str(e)}")
                    continue
        
        self.date_columns = date_columns
        return df
    
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Load CSV or Excel file with automatic date detection"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, low_memory=False)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded file: {file_path} ({len(self.df)} rows, {len(self.df.columns)} columns)")
            
            self.df = self._auto_detect_dates(self.df)
            self._handle_missing_values()
            self._analyze_columns()
            
            # Convert preview data to JSON-serializable format
            preview_df = self.df.head(5).copy()
            for col in preview_df.columns:
                if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                    preview_df[col] = preview_df[col].astype(str)

            return {
                "columns": self.df.columns.tolist(),
                "row_count": len(self.df),
                "column_count": len(self.df.columns),
                "date_columns": self.date_columns,
                "preview": preview_df.to_dict('records'),
                "column_info": self.column_info,
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ValueError(f"Data loading error: {str(e)}")
    
    def _handle_missing_values(self):
        """Intelligently handle missing values"""
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self.df[col] = self.df[col].ffill()
                else:
                    mode_value = self.df[col].mode()
                    if len(mode_value) > 0:
                        self.df[col] = self.df[col].fillna(mode_value[0])
                    else:
                        self.df[col] = self.df[col].fillna('Unknown')
    
    def _analyze_columns(self):
        """Deep analysis of column characteristics"""
        self.column_info = {}
        
        for col in self.df.columns:
            col_data = self.df[col]
            
            # Convert sample values to JSON-serializable format
            sample_values = col_data.head(5)
            if pd.api.types.is_datetime64_any_dtype(col_data):
                sample_values = [str(x) if pd.notna(x) else None for x in sample_values]
            else:
                sample_values = sample_values.tolist()
            
            info = {
                "dtype": str(col_data.dtype),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(col_data.isnull().sum() / len(col_data) * 100),
                "unique_count": int(col_data.nunique()),
                "sample_values": sample_values
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                info["role"] = "numeric"
                info["stats"] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std())
                }
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                info["role"] = "datetime"
                info["stats"] = {
                    "min_date": str(col_data.min()),
                    "max_date": str(col_data.max()),
                    "range_days": (col_data.max() - col_data.min()).days if col_data.notna().any() else 0
                }
            else:
                info["role"] = "categorical"
                top_values = col_data.value_counts().head(10)
                info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            self.column_info[col] = info
    
    def analyze_query(self, query: str, use_search: bool = True) -> Dict[str, Any]:
        """Analyze query using Gemini 2.5 Flash with optional Google Search"""
        if self.df is None:
            return {"error": "No data loaded. Please upload a dataset first."}
        
        try:
            needs_search = self._query_needs_search(query)
            should_search = use_search and needs_search
            
            query_analysis = self._analyze_query_with_gemini(query, should_search)
            
            if "error" in query_analysis:
                return query_analysis
            
            result = self._execute_data_processing(query_analysis)
            return result
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _query_needs_search(self, query: str) -> bool:
        """Determine if query benefits from web search"""
        search_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'trend',
            'compare with', 'industry', 'benchmark', 'average',
            'what is', 'who is', 'when did', 'how does'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in search_keywords)
    
    def _analyze_query_with_gemini(self, query: str, use_search: bool) -> Dict[str, Any]:
        """Use Gemini 2.5 Flash with structured outputs"""
        
        # Convert DataFrame to JSON-serializable format
        sample_df = self.df.head(10).copy()
        for col in sample_df.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                sample_df[col] = sample_df[col].astype(str)

        data_context = {
            "columns": list(self.column_info.keys()),
            "column_details": self.column_info,
            "row_count": len(self.df),
            "date_columns": self.date_columns,
            "sample_data": sample_df.to_dict('records')
        }
        
        prompt = f"""You are an expert data analyst. Analyze this query and dataset to create a precise analysis plan.

Dataset Context:
{json.dumps(data_context, indent=2)}

User Query: "{query}"

Analyze:
1. What type of analysis is needed
2. Which columns are relevant
3. What aggregation function to use (sum, mean, count, max, min)
4. How to group the data
5. What chart type is appropriate (bar, line, pie, scatter, area, doughnut)
6. Sort order and limits

Return a complete analysis plan as JSON."""

        try:
            config_dict = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string"},
                    "group_by_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "aggregation_function": {"type": "string"},
                    "chart_type": {"type": "string"},
                    "title": {"type": "string"},
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "sort_order": {"type": "string"},
                    "limit_results": {"type": "integer"},
                    "date_grouping": {"type": "string"},
                    "interpretation": {"type": "string"}
                },
                "required": ["analysis_type", "value_column", "chart_type", "interpretation"]
            }
        }
            
            tools = []
# IMPORTANT: Gemini doesn't support tools + JSON response together
# So we do TWO API calls when search is needed
            if use_search:
                logger.info("Using Google Search grounding for context")
                
                # First call: Get context with search (no JSON schema)
                search_response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"Based on this query, provide relevant context: {query}",
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )
                
                # Add search context to the prompt
                search_context = search_response.text
                prompt = f"""Search Context: {search_context}

            {prompt}"""
                
            # Second call: Get structured JSON response (no tools)
            config = types.GenerateContentConfig(**config_dict)
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            result = json.loads(response.text)
            logger.info(f"Query analyzed: {result.get('analysis_type', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {"error": f"AI analysis failed: {str(e)}"}
    
    def _execute_data_processing(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pandas operations based on Gemini's analysis"""
        
        try:
            group_by = analysis.get("group_by_column")
            value_col = analysis.get("value_column")
            agg_func = analysis.get("aggregation_function", "sum")
            sort_order = analysis.get("sort_order", "desc")
            limit = analysis.get("limit_results", 10)
            date_grouping = analysis.get("date_grouping")
            
            if value_col not in self.df.columns:
                return {"error": f"Column '{value_col}' not found"}
            
            if group_by and group_by not in self.df.columns:
                return {"error": f"Column '{group_by}' not found"}
            
            df_filtered = self.df.copy()
            filters = analysis.get("filters", {})
            if filters:
                df_filtered = self._apply_filters(df_filtered, filters)
            
            if date_grouping and group_by in self.date_columns:
                df_filtered = self._add_date_grouping(df_filtered, group_by, date_grouping)
                group_by = f"{group_by}_{date_grouping}"
            
            if group_by:
                agg_functions = {
                    "sum": lambda x: x.sum(),
                    "mean": lambda x: x.mean(),
                    "average": lambda x: x.mean(),
                    "count": lambda x: x.count(),
                    "max": lambda x: x.max(),
                    "min": lambda x: x.min(),
                }
                
                func = agg_functions.get(agg_func.lower(), lambda x: x.sum())
                result_data = df_filtered.groupby(group_by)[value_col].apply(func)
                
            else:
                if agg_func == "max":
                    idx = df_filtered[value_col].idxmax()
                    result_data = pd.Series([df_filtered.loc[idx, value_col]], 
                                          index=[df_filtered.loc[idx, self.df.columns[0]]])
                elif agg_func == "min":
                    idx = df_filtered[value_col].idxmin()
                    result_data = pd.Series([df_filtered.loc[idx, value_col]], 
                                          index=[df_filtered.loc[idx, self.df.columns[0]]])
                else:
                    result_data = df_filtered[value_col].head(limit)
            
            if isinstance(result_data, pd.Series):
                if sort_order == "desc":
                    result_data = result_data.sort_values(ascending=False)
                elif sort_order == "asc":
                    result_data = result_data.sort_values(ascending=True)
                
                result_data = result_data.head(limit)
            
            labels = [str(x) for x in result_data.index.tolist()]
            values = [float(x) if pd.notna(x) else 0 for x in result_data.values.tolist()]
            
            chart_config = {
                "chartType": analysis.get("chart_type", "bar"),
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": analysis.get("y_label", value_col),
                        "data": values,
                        "backgroundColor": self._generate_colors(len(values))
                    }]
                },
                "config": {
                    "title": analysis.get("title", "Data Analysis"),
                    "xAxisLabel": analysis.get("x_label", group_by or "Category"),
                    "yAxisLabel": analysis.get("y_label", value_col)
                },
                "interpretation": analysis.get("interpretation", "Analysis complete"),
                "statistics": {
                    "total": float(sum(values)),
                    "average": float(np.mean(values)),
                    "max": float(max(values)) if values else 0,
                    "min": float(min(values)) if values else 0,
                    "count": len(values)
                },
                "processingSteps": [
                    f"Dataset: {len(self.df)} rows",
                    f"Grouped by: {group_by if group_by else 'None'}",
                    f"Aggregation: {value_col} using {agg_func}",
                    f"Sorted: {sort_order}",
                    f"Results: Top {len(values)}"
                ]
            }
            
            logger.info(f"Processing complete: {len(values)} results")
            return chart_config
            
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        for col, condition in filters.items():
            if col in df.columns:
                if isinstance(condition, dict):
                    if "gt" in condition:
                        df = df[df[col] > condition["gt"]]
                    if "lt" in condition:
                        df = df[df[col] < condition["lt"]]
                    if "eq" in condition:
                        df = df[df[col] == condition["eq"]]
        return df
    
    def _add_date_grouping(self, df: pd.DataFrame, date_col: str, grouping: str) -> pd.DataFrame:
        """Add date-based grouping column"""
        new_col = f"{date_col}_{grouping}"
        
        if grouping == "year":
            df[new_col] = df[date_col].dt.year
        elif grouping == "month":
            df[new_col] = df[date_col].dt.to_period('M').astype(str)
        elif grouping == "quarter":
            df[new_col] = df[date_col].dt.to_period('Q').astype(str)
        elif grouping == "week":
            df[new_col] = df[date_col].dt.to_period('W').astype(str)
        elif grouping == "day":
            df[new_col] = df[date_col].dt.date
        
        return df
    
    def _generate_colors(self, count: int) -> List[str]:
        """Generate color palette"""
        colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
            '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384',
            '#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6'
        ]
        return [colors[i % len(colors)] for i in range(count)]