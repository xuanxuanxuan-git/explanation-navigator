import React from "react"
import "./typingStatus.css"

const STATUS = {
  thinking: {
    text: "Thinking…",
    // color: "#6b7280"
  },
  retrieving: {
    text: "Retrieving explanations…",
    // color: "#2563eb"
  },
  rephrasing: {
    text: "Writing answer…",
    // color: "#10b981"
  }
}

export default function TypingStatus({ status = "thinking" }) {

  const config = STATUS[status] || STATUS.thinking

  return (
    <div
      className="typing-status"
      style={{ color: "#6b7280" }}
    >
      {config.text}
    </div>
  )
}