# AI Chatbot Project

A simple chatbot application with a React frontend and Python FastAPI backend.

## Project Structure

```
Chatbot - Backend/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # FastAPI server
│   ├── project.py          # Chatbot logic
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatMessage.js
│   │   │   └── MessageInput.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend folder:

   ```bash
   cd backend
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   python main.py
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend folder:

   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## Features

- Clean and simple chat interface
- Real-time messaging
- Responsive design with Tailwind CSS
- Loading indicators
- Error handling
- CORS enabled for local development

## Usage

1. Start both the backend and frontend servers
2. Open your browser to `http://localhost:3000`
3. Start chatting with the AI assistant!

## Technologies Used

- **Frontend**: React, Tailwind CSS
- **Backend**: Python, FastAPI, LangChain, HuggingFace
- **AI Model**: HuggingFace GPT model
