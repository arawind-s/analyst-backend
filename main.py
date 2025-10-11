from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
from src.agent import DataAnalysisAgent
import sys
sys.path.append('src') 

app = FastAPI(title="Data Analysis Dashboard API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Agent instance
agent = DataAnalysisAgent()

@app.get("/")
def read_root():
    return {
        "message": "Data Analysis Dashboard API",
        "endpoints": {
            "upload": "POST /upload - Upload dataset (CSV/Excel)",
            "query": "POST /query - Ask question about data",
            "health": "GET /health - Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "agent": "ready"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load data into agent
        data_info = agent.load_data(str(file_path))
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "data_info": data_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(query: str = Form(...)):
    """
    Ask a question about the uploaded data
    Body (form-data):
    - query: Your question about the data
    """
    try:
        if agent.df is None:
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first using /upload endpoint"
            )
        
        result = agent.analyze_query(query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Add this test endpoint to main.py
@app.get("/data-info")
def get_data_info():
    if agent.df is None:
        return {"error": "No data loaded"}
    
    return {
        "columns": agent.df.columns.tolist(),
        "dtypes": agent.df.dtypes.astype(str).to_dict(),
        "sample": agent.df.head(10).to_dict('records'),
        "shape": agent.df.shape,
        "null_counts": agent.df.isnull().sum().to_dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)