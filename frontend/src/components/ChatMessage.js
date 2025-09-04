import React from "react";

const ChatMessage = ({ message, isUser, timestamp, isSystem = false }) => {
  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (isSystem) {
    return (
      <div className="flex justify-center mb-4">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 max-w-md">
          <div className="flex items-center space-x-2">
            <span className="text-yellow-600">ℹ️</span>
            <div className="flex-1">
              <p className="text-sm text-yellow-800 whitespace-pre-wrap">
                {message}
              </p>
              <p className="text-xs text-yellow-600 mt-1">
                {formatTime(timestamp)}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Format message text for better readability
  const formatText = (text) => {
    // Split text into paragraphs and handle line breaks
    return text
      .split("\n")
      .map((paragraph, index) => {
        if (paragraph.trim() === "") return null;
        return (
          <p key={index} className={index > 0 ? "mt-2" : ""}>
            {paragraph.trim()}
          </p>
        );
      })
      .filter(Boolean);
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg shadow-sm ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-white text-gray-800 border border-gray-200"
        }`}
      >
        <div className="text-sm whitespace-pre-wrap leading-relaxed">
          {isUser ? (
            <p>{message}</p>
          ) : (
            <div className="space-y-1">{formatText(message)}</div>
          )}
        </div>
        <p
          className={`text-xs mt-2 ${
            isUser ? "text-blue-100" : "text-gray-500"
          }`}
        >
          {formatTime(timestamp)}
        </p>
      </div>
    </div>
  );
};

export default ChatMessage;
