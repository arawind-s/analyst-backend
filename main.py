from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from src.Analyst import DataAnalysisAgent
from src.database import init_db, get_db, User, Conversation, Message, Dataset
from src.auth import get_password_hash, verify_password, create_access_token, get_current_user
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Data Analysis Dashboard API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Agent instance
agent = DataAnalysisAgent()

# Pydantic models
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class CreateConversationRequest(BaseModel):
    title: str = "New Chat"

# ============= AUTH ENDPOINTS =============

@app.post("/signup")
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        id=str(uuid.uuid4()),
        email=request.email,
        hashed_password=get_password_hash(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create token
    access_token = create_access_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email
        }
    }

@app.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email
        }
    }

@app.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat()
    }

# ============= CONVERSATION ENDPOINTS =============

@app.post("/conversations")
async def create_conversation(
    request: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        title=request.title
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat()
    }

@app.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()
    
    return {
        "conversations": [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
                "message_count": len(c.messages)
            }
            for c in conversations
        ]
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "chart_data": m.chart_data,
                "timestamp": m.timestamp.isoformat()
            }
            for m in conversation.messages
        ]
    }

# ============= FILE UPLOAD ENDPOINT =============

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        dataset_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{dataset_id}_{file.filename}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load data into agent
        data_info = agent.load_data(str(file_path))
        
        # Save dataset metadata
        dataset = Dataset(
            id=dataset_id,
            user_id=current_user.id,
            filename=file.filename,
            storage_path=str(file_path),
            row_count=data_info['row_count'],
            columns=data_info['columns'],
            dtypes=data_info.get('dtypes', {})
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return {
            "message": "File uploaded successfully",
            "dataset_id": dataset_id,
            "filename": file.filename,
            "data_info": data_info
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ============= QUERY ENDPOINT WITH CHAT HISTORY =============

@app.post("/query")
async def query_data(
    query: str = Form(...),
    conversation_id: str = Form(...),
    dataset_id: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify conversation belongs to user
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Save user message
        user_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=query
        )
        db.add(user_message)
        
        # Load dataset if provided
        if dataset_id:
            dataset = db.query(Dataset).filter(
                Dataset.id == dataset_id,
                Dataset.user_id == current_user.id
            ).first()
            
            if dataset:
                agent.load_data(dataset.storage_path)
        
        # Check if data is loaded
        if agent.df is None:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first")
        
        # Analyze query
        result = agent.analyze_query(query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Save assistant message
        assistant_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=result.get("interpretation", "Analysis complete"),
            chart_data=result
        )
        db.add(assistant_message)
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        
        # Auto-update conversation title if it's the first message
        if len(conversation.messages) == 0:
            conversation.title = query[:50] + "..." if len(query) > 50 else query
        
        db.commit()
        
        return {
            **result,
            "message_id": assistant_message.id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "message": "Data Analysis Dashboard API with Authentication",
        "endpoints": {
            "signup": "POST /signup",
            "login": "POST /login",
            "me": "GET /me",
            "conversations": "GET /conversations",
            "upload": "POST /upload",
            "query": "POST /query"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "agent": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)