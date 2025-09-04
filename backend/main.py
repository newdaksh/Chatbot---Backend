from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from project import (
    chat, 
    clear_conversation_history, 
    get_conversation_summary, 
    process_pdf, 
    clear_pdf_data, 
    get_pdf_info,
    preload_global_pdfs,  # NEW
)
import uuid

app = FastAPI(title="Chatbot API with PDF Support", description="A chatbot API with memory and PDF processing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionResponse(BaseModel):
    message: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Generate session ID if not provided
        if not request.session_id or request.session_id == "default":
            session_id = str(uuid.uuid4())
        else:
            session_id = request.session_id
            
        response = chat(request.message, session_id=session_id)
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history", response_model=SessionResponse)
async def clear_history_endpoint(session_id: str = "default"):
    try:
        message = clear_conversation_history(session_id)
        return SessionResponse(message=message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-summary", response_model=SessionResponse)
async def get_summary_endpoint(session_id: str = "default"):
    try:
        summary = get_conversation_summary(session_id)
        return SessionResponse(message=summary, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/new-session", response_model=SessionResponse)
async def new_session_endpoint():
    try:
        new_session_id = str(uuid.uuid4())
        return SessionResponse(message="New session created", session_id=new_session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf", response_model=SessionResponse)
async def upload_pdf_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        file_content = await file.read()
        
        # Process PDF
        result = process_pdf(file_content, file.filename, session_id)
        
        return SessionResponse(message=result, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-pdf", response_model=SessionResponse)
async def clear_pdf_endpoint(session_id: str = "default"):
    try:
        message = clear_pdf_data(session_id)
        return SessionResponse(message=message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf-info", response_model=SessionResponse)
async def get_pdf_info_endpoint(session_id: str = "default"):
    try:
        info = get_pdf_info(session_id)
        return SessionResponse(message=info, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Chatbot API with PDF Support is running!"}

# NEW: preload PDFs at app startup
@app.on_event("startup")
async def preload_pdfs_startup():
    msg = preload_global_pdfs()  # loads from backend/pdfs by default
    print(msg)

# OPTIONAL: endpoint to reload preloaded PDFs without restarting
@app.post("/reload-preloaded-pdfs", response_model=SessionResponse)
async def reload_preloaded_pdfs_endpoint(pdf_dir: str = "pdfs"):
    try:
        msg = preload_global_pdfs(pdf_dir)
        return SessionResponse(message=msg, session_id="global")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
