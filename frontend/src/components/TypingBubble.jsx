import React from 'react'
import './typingBubble.css'

export default function TypingBubble() {
  return (
    <div
      className="typing-wrap"
      aria-live="polite"
      aria-busy="true"
    >
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  )
}