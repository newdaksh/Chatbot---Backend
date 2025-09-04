import React, { useState, useEffect, useRef } from "react";
import ChatMessage from "./components/ChatMessage";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [uploadedPdf, setUploadedPdf] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Create new session on component mount
  useEffect(() => {
    createNewSession();
  }, []);

  // Scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const createNewSession = async () => {
    try {
      const response = await fetch("http://localhost:8000/new-session");
      const data = await response.json();
      setSessionId(data.session_id);
      setMessages([]);
      setUploadedPdf(null);
    } catch (error) {
      console.error("Error creating new session:", error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please select a PDF file only.");
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("session_id", sessionId);

      const response = await fetch("http://localhost:8000/upload-pdf", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadedPdf({
          name: file.name,
          size: file.size,
          uploadTime: new Date(),
        });

        // Add system message about PDF upload
        const systemMessage = {
          id: Date.now(),
          text: data.message,
          isUser: false,
          timestamp: new Date(),
          isSystem: true,
        };
        setMessages((prev) => [...prev, systemMessage]);
      } else {
        throw new Error(data.detail || "Upload failed");
      }
    } catch (error) {
      alert("Error uploading PDF: " + error.message);
    } finally {
      setIsUploading(false);
      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: inputMessage,
          session_id: sessionId,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          id: Date.now() + 1,
          text: data.response,
          isUser: false,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);

        // Update session ID if it changed
        if (data.session_id !== sessionId) {
          setSessionId(data.session_id);
        }
      } else {
        throw new Error(data.detail || "Something went wrong");
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = async () => {
    try {
      await fetch(
        `http://localhost:8000/clear-history?session_id=${sessionId}`,
        {
          method: "POST",
        }
      );
      setMessages([]);
    } catch (error) {
      console.error("Error clearing history:", error);
    }
  };

  const clearPdf = async () => {
    try {
      await fetch(`http://localhost:8000/clear-pdf?session_id=${sessionId}`, {
        method: "POST",
      });
      setUploadedPdf(null);

      const systemMessage = {
        id: Date.now(),
        text: "PDF data has been cleared. You can upload a new document.",
        isUser: false,
        timestamp: new Date(),
        isSystem: true,
      };
      setMessages((prev) => [...prev, systemMessage]);
    } catch (error) {
      console.error("Error clearing PDF:", error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 shadow-lg">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold">AI Personal Advisor</h1>
            <p className="text-sm text-blue-100">
              Session:{" "}
              {sessionId ? sessionId.substring(0, 8) + "..." : "Loading..."}
              {uploadedPdf && (
                <span className="ml-2 px-2 py-1 bg-green-500 rounded text-xs">
                  📄 PDF Loaded
                </span>
              )}
            </p>
          </div>
          <div className="flex gap-2">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              ref={fileInputRef}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="px-3 py-1 bg-green-500 hover:bg-green-400 rounded text-sm transition-colors disabled:opacity-50"
            >
              {isUploading ? "Uploading..." : "📄 Upload PDF"}
            </button>
            {uploadedPdf && (
              <button
                onClick={clearPdf}
                className="px-3 py-1 bg-red-500 hover:bg-red-400 rounded text-sm transition-colors"
              >
                Clear PDF
              </button>
            )}
            <button
              onClick={clearHistory}
              className="px-3 py-1 bg-blue-500 hover:bg-blue-400 rounded text-sm transition-colors"
            >
              Clear History
            </button>
            <button
              onClick={createNewSession}
              className="px-3 py-1 bg-green-500 hover:bg-green-400 rounded text-sm transition-colors"
            >
              New Session
            </button>
          </div>
        </div>

        {/* PDF Info Bar */}
        {uploadedPdf && (
          <div className="mt-2 p-2 bg-blue-500 rounded text-sm">
            <div className="flex items-center justify-between">
              <span>
                📄 {uploadedPdf.name} ({formatFileSize(uploadedPdf.size)})
              </span>
              <span className="text-blue-100">
                Uploaded at {uploadedPdf.uploadTime.toLocaleTimeString()}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <div className="mb-4">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🤖</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                Welcome to AI Personal MEDICAL Advisor!
              </h3>
              <p className="text-sm text-gray-500 max-w-md mx-auto mb-4">
                I'm here to help you with advice, questions, and guidance. I'll
                remember our conversation context for better assistance.
              </p>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 max-w-md mx-auto">
                <h4 className="font-medium text-green-800 mb-2">
                  📄 PDF Support
                </h4>
                <p className="text-sm text-green-700">
                  Upload a PDF document and I'll answer questions based on its
                  content using advanced vector search!
                </p>
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message.text}
              isUser={message.isUser}
              timestamp={message.timestamp}
            />
          ))
        )}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg p-4 shadow-sm border max-w-xs">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.1s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                </div>
                <span className="text-sm text-gray-500">
                  {uploadedPdf ? "Searching PDF..." : "AI is thinking..."}
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <div className="flex space-x-4 max-w-4xl mx-auto">
          <div className="flex-1 relative">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                uploadedPdf
                  ? "Ask me anything about the uploaded PDF..."
                  : "Type your message... (I'll remember our conversation)"
              }
              className="w-full p-3 pr-12 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows="1"
              style={{ minHeight: "44px", maxHeight: "120px" }}
              disabled={isLoading || isUploading}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading || isUploading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              "Send"
            )}
          </button>
        </div>
        <div className="text-center text-xs text-gray-400 mt-2">
          {uploadedPdf ? (
            <>� Vector search enabled • �💡 Context memory active</>
          ) : (
            <>
              💡 I remember the last 5 message pairs • 📄 Upload PDF for
              document-based chat
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
