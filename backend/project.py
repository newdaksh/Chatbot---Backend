import re
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
from typing import Optional
import glob

# Setup HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    HF_TOKEN = os.getenv("HF_TOKEN")
)

model = ChatHuggingFace(llm=llm)

# Setup embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store conversation history and vector stores per session
conversation_memory = {}
vector_stores = {}
pdf_contents = {}

# NEW: global KB vector store (preloaded PDFs)
global_vector_store = None
preloaded_pdf_info = {"filenames": [], "num_pages": 0, "num_chunks": 0}

faiss_path = os.path.join(os.path.dirname(__file__), "faiss_indexes/global_kb")

if os.path.exists(faiss_path):
    try:
        global_vector_store = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True  # 👈 add this
        )
        print("✅ Global FAISS index loaded from disk")
    except Exception as e:
        print(f"⚠️ Could not load FAISS index: {e}")

# Define your system prompt
system_prompt = """You are a helpful Personal Advisor. Please provide clear, concise, and direct answers to user questions. 
Always introduce yourself as 'Your Personal AI Medical Advisor' or 'AI Personal Advisor' whenever asked about your identity. Never call yourself ChatGPT, 
OpenAI AI, or any other name.

When a PDF document has been uploaded:
- Use the information from the PDF to answer questions
- Be specific and reference the document content
- If the question cannot be answered from the PDF, mention that the information is not available in the uploaded document
- Always be accurate to the source material

Guidelines:
- Give natural, conversational responses
- Be helpful and informative
- Keep answers focused and relevant
- Use simple, easy-to-understand language
- Avoid unnecessary formatting or prefixes
- Provide practical advice when appropriate
- Remember the conversation context and refer to previous messages when relevant

1. ROLE & IDENTITY
   - Act as a professional, knowledgeable, and supportive personal Medical advisor.
   - Maintain a balance between being informative, approachable, and respectful.

2. COMMUNICATION STYLE
   - Use clear, concise, and polite language.
   - Adapt tone depending on context: professional for technical queries, friendly yet respectful for casual advice.
   - Avoid unnecessary jargon, but explain technical terms when required.
   - Provide step-by-step explanations when guiding users through processes.

3. CAPABILITIES
   - Answer technical, educational, and general advice queries with correctness and context-awareness.
   - Provide structured responses: (a) short summary first, (b) detailed explanation if needed, (c) actionable suggestions.
   - Use examples, analogies, or best practices where beneficial.
   - Handle ambiguity by asking clarifying questions instead of making assumptions.

4. BOUNDARIES
   - Do not provide harmful, unsafe, or unethical advice.
   - Do not guess sensitive personal details about the user.
   - If unsure about something, state the uncertainty clearly instead of fabricating information.

5. RESPONSE FORMAT
   - Be concise, but expand when depth is required.
   - Use bullet points, numbered steps, or short paragraphs for readability.
   - Always remain professional, polite, and empathetic.
   - Keep your response between 25 and 40 words maximum.
   - If the response exceeds 40 words, summarize and condense it to fit within the limit.

Answer the user's question directly and helpfully ."""


# Define sequential question flows
question_flows = {
    "medical_checkup": [
        "What is your age?",
        "Do you have any existing medical conditions?",
        "Are you currently taking any medications?",
        "Do you have any allergies?",
        "What symptoms are you experiencing?"
    ]
}

# Track per-session flow progress
user_flows = {}  # {session_id: {"flow": "medical_checkup", "step": 0, "answers": []}}


def process_pdf(file_content: bytes, filename: str, session_id: str = "default") -> str:
    """Process PDF file and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        vectorstore.save_local(f"faiss_indexes/{session_id}")

        
        # Store vector store for this session
        vector_stores[session_id] = vectorstore
        
        # Store PDF info
        pdf_contents[session_id] = {
            'filename': filename,
            'num_pages': len(documents),
            'num_chunks': len(splits)
        }

        # Clean up temp file
        os.unlink(temp_file_path)

        return f"PDF '{filename}' processed successfully! {len(documents)} pages, {len(splits)} text chunks created. You can now ask questions about this document."

    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def preload_global_pdfs(pdf_dir: Optional[str] = None) -> str:
    """Preload PDFs from a directory into a global vector store available to all sessions."""
    import traceback
    global global_vector_store, preloaded_pdf_info

    try:
        if pdf_dir is None:
            pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
        pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if not pdf_paths:
            global_vector_store = None
            preloaded_pdf_info = {"filenames": [], "num_pages": 0, "num_chunks": 0}
            return f"No PDFs found to preload in: {pdf_dir}"

        all_docs = []
        total_pages = 0
        filenames = []

        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            total_pages += len(docs)
            for d in docs:
                # Keep track of source file in metadata
                d.metadata = {**(d.metadata or {}), "source": os.path.basename(path)}
            all_docs.extend(docs)
            filenames.append(os.path.basename(path))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(all_docs)

        # Build/replace the global vector store
        global_vector_store = FAISS.from_documents(splits, embeddings)
        preloaded_pdf_info = {
            "filenames": filenames,
            "num_pages": total_pages,
            "num_chunks": len(splits),
        }
        
        # Save to disk
        save_path = os.path.join(os.path.dirname(__file__), "faiss_indexes/global_kb")
        os.makedirs(save_path, exist_ok=True)
        
        global_vector_store.save_local("faiss_indexes/global_kb")
        return (
            f"Preloaded {len(filenames)} PDFs into knowledge base: "
            f"{', '.join(filenames)} | {total_pages} pages, {len(splits)} chunks."
        )
    except Exception as e:
        global_vector_store = None
        preloaded_pdf_info = {"filenames": [], "num_pages": 0, "num_chunks": 0}
        return f"Error preloading PDFs: {str(e)}"

def get_conversation_history(session_id="default", max_messages=5):
    """Get the last 5 messages from conversation history"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    # Return last max_messages pairs (user + assistant messages)
    history = conversation_memory[session_id]
    return history[-max_messages*2:] if len(history) > max_messages*2 else history

def add_to_conversation_history(session_id="default", user_message="", assistant_message=""):
    """Add messages to conversation history"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    conversation_memory[session_id].extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ])
    
    # Keep only last 10 messages (5 pairs)
    if len(conversation_memory[session_id]) > 10:
        conversation_memory[session_id] = conversation_memory[session_id][-10:]

def format_conversation_context(history):
    """Format conversation history for the prompt"""
    if not history:
        return ""
    
    context = "\n\nPrevious conversation context:\n"
    for msg in history:
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            context += f"Assistant: {msg['content']}\n"
    
    context += "\nCurrent conversation:\n"
    return context

def search_pdf_content(query: str, session_id: str = "default", k: int = 3) -> str:
    """Search both preloaded KB PDFs and (if present) the session's uploaded PDF using vector similarity."""
    contexts = []

    # Search preloaded global KB
    try:
        if global_vector_store is not None:
            kb_docs = global_vector_store.similarity_search(query, k=k)
            if kb_docs:
                context = "\n\nRelevant information from the knowledge base (preloaded PDFs):\n"
                for i, doc in enumerate(kb_docs, 1):
                    src = (doc.metadata or {}).get("source", "preloaded PDF")
                    context += f"[KB {i} - {src}]: {doc.page_content[:500]}...\n\n"
                contexts.append(context)
    except Exception as e:
        contexts.append(f"\nError searching knowledge base: {str(e)}\n")

    # Search session-specific uploaded PDF
    if session_id in vector_stores:
        try:
            vectorstore = vector_stores[session_id]
            docs = vectorstore.similarity_search(query, k=k)
            if docs:
                pdf_info = pdf_contents.get(session_id, {})
                name = pdf_info.get('filename', 'uploaded PDF')
                context = f"\n\nDocument Context (from '{name}'):\n"
                for i, doc in enumerate(docs, 1):
                    context += f"[Excerpt {i}]: {doc.page_content[:500]}...\n\n"
                contexts.append(context)
        except Exception as e:
            contexts.append(f"\nError searching session PDF: {str(e)}\n")

    return "".join(contexts)


def clean_response(raw_response):
    """Clean and format the AI response with proper markdown formatting"""
    import re
    
    # Remove common unwanted prefixes and suffixes
    response = raw_response.strip()
    
    # Replace undesired identity mentions
    response = response.replace("ChatGPT", "AI Assistant, your Personal Advisor")
    response = response.replace("OpenAI", "AI Assistant, your Personal Advisor")
    
    # Remove quotes if the response is wrapped in them
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]
    
    # Remove "Short answer:" or similar prefixes
    prefixes_to_remove = [
        "short answer:",
        "answer:",
        "response:",
        "**short answer**",
        "**answer**",
        "assistant:",
        "ai:"
    ]
    
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Clean up HTML-like tags but preserve markdown
    response = re.sub(r'<(?!/?(?:br|p|div|span)\b)[^>]+>', '', response)
    
    # Normalize line breaks and preserve paragraph structure
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # Multiple line breaks to double
    # Better: preserve markdown-friendly line breaks
    response = re.sub(r'\n{3,}', '\n\n', response)  # Collapse 3+ newlines into 2
    response = re.sub(r'[ \t]+\n', '\n', response)  # Remove trailing spaces before newline
    response = re.sub(r'\n[ \t]+', '\n', response)  # Remove indentation before newline
    
    # Enhance markdown formatting
    # Convert numbered lists if not already formatted
    response = re.sub(r'(?:^|\n)(\d+\.)\s*([^\n]+)', r'\n\1 \2', response)
    
    # Convert bullet points if not already formatted
    response = re.sub(r'(?:^|\n)[-•]\s*([^\n]+)', r'\n- \1', response)
    
    # Remove markdown headers (###, ##, #) but keep the text
    response = re.sub(r'(?:^|\n)#{1,6}\s*([^\n]+)', r'\n\1\n', response)
    
    # Ensure proper bold/italic formatting (don't remove, enhance)
    # Fix incomplete bold formatting
    response = re.sub(r'\*([^*\n]+)\*(?!\*)', r'*\1*', response)  # Ensure italic
    response = re.sub(r'\*\*([^*\n]+?)\*\*', r'**\1**', response)  # Ensure bold
    
    # Add code block formatting for technical terms if needed
    # This is optional - you can customize based on your needs
    technical_terms = ['API', 'HTTP', 'JSON', 'SQL', 'PDF', 'URL', 'HTML', 'CSS', 'JavaScript']
    for term in technical_terms:
        response = re.sub(rf'\b{term}\b(?!`)', f'`{term}`', response)
    
    # Clean up excessive whitespace while preserving markdown structure
    response = re.sub(r' +', ' ', response)  # Multiple spaces to single
    response = response.strip()
    
    # Ensure proper paragraph ending
    if response and not response[-1] in '.!?':
        # Only add period if the last line isn't a header or list item
        last_line = response.split('\n')[-1].strip()
        if not (last_line.startswith('#') or last_line.startswith('-') or last_line.startswith('*') or 
                re.match(r'^\d+\.', last_line) or last_line.endswith(':')):
            response += '.'
    
    return response

def chat(message, history=None, session_id="default"):
    
    # 1. Check if user is already in a flow
    if session_id in user_flows:
        flow_data = user_flows[session_id]
        flow_name = flow_data["flow"]

        # Save last answer
        flow_data["answers"].append(message)
        flow_data["step"] += 1

        # If more questions remain
        if flow_data["step"] < len(question_flows[flow_name]):
            next_q = question_flows[flow_name][flow_data["step"]]
            return next_q
        else:
            # Flow finished → summarize answers
            summary = "\n".join(
                f"{q} {a}" for q, a in zip(question_flows[flow_name], flow_data["answers"])
            )
            del user_flows[session_id]
            return f"✅ Thanks! Here’s a summary of your responses:\n{summary}"

    # 2. Trigger a flow if user types the start command
    triggers = ["medical checkup", "start checkup", "begin checkup"]
    if any(t in message.lower() for t in triggers):
        user_flows[session_id] = {"flow": "medical_checkup", "step": 0, "answers": []}
        return question_flows["medical_checkup"][0]
    
    
    
    try:
        # Get conversation history
        conversation_history = get_conversation_history(session_id)
        
        # Format the conversation context
        context = format_conversation_context(conversation_history)
        
        # Search PDF content if available (both global KB and session-specific)
        pdf_context = search_pdf_content(message, session_id)
        
        # Combine system prompt, context, PDF context, and current message
        full_message = f"{system_prompt}{context}{pdf_context}User: {message}\nAssistant:"
        
        # Get response from the model
        response = model.invoke(full_message)
        
        # Clean and format the response
        cleaned_response = clean_response(response.content)

        
        
        # If the response is too short or seems incomplete, try a different approach
        if len(cleaned_response) < 10:
            # Fallback with a simpler prompt
            simple_prompt = f"{context}{pdf_context}Please provide a helpful and clear answer to: {message}"
            response = model.invoke(simple_prompt)
            cleaned_response = clean_response(response.content)
        
        # Add to conversation history
        add_to_conversation_history(session_id, message, cleaned_response)
        
        return cleaned_response
        
    except Exception as e:
        error_message = "I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question."
        # Still add to history even on error
        add_to_conversation_history(session_id, message, error_message)
        return error_message

def clear_conversation_history(session_id="default"):
    """Clear conversation history for a session"""
    if session_id in conversation_memory:
        conversation_memory[session_id] = []
    return "Conversation history cleared!"

def clear_pdf_data(session_id="default"):
    """Clear PDF data for a session"""
    if session_id in vector_stores:
        del vector_stores[session_id]
    if session_id in pdf_contents:
        del pdf_contents[session_id]
    return "PDF data cleared!"

def get_conversation_summary(session_id="default"):
    """Get a summary of the current conversation"""
    history = get_conversation_history(session_id)
    if not history:
        return "No conversation history found."
    
    summary = f"Conversation contains {len(history)} messages:\n"
    for i, msg in enumerate(history, 1):
        role = "You" if msg["role"] == "user" else "Assistant"
        preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        summary += f"{i}. {role}: {preview}\n"
    
    return summary

def get_pdf_info(session_id="default"):
    """Get information about preloaded PDFs and the session's uploaded PDF."""
    parts = []
    if preloaded_pdf_info.get("filenames"):
        parts.append(
            f"KB: {len(preloaded_pdf_info['filenames'])} PDFs "
            f"({', '.join(preloaded_pdf_info['filenames'])}); "
            f"{preloaded_pdf_info['num_pages']} pages, {preloaded_pdf_info['num_chunks']} chunks"
        )

    if session_id in pdf_contents:
        info = pdf_contents[session_id]
        parts.append(
            f"Session PDF: '{info['filename']}' - {info['num_pages']} pages, {info['num_chunks']} chunks"
        )

    if not parts:
        return "No PDF knowledge loaded."

    return " | ".join(parts)