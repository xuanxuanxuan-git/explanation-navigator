import React, { useState } from 'react'
import ChatPage from './components/ChatPage.jsx'

export default function App() {

  return (
    <div style={{ fontFamily: 'system-ui, Arial', padding: 16 }}>
      <h2>LLM Chatbot</h2>
      <ChatPage />
    </div>
  )
}
