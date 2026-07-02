import React, { useState } from 'react'
import ChatPage from './components/ChatPage.jsx'

export default function App() {

  return (
    <div style={{ fontFamily: 'system-ui, Arial', padding: 16, height: "87h" }}>
      <h2 style={{ marginTop: 10, marginBottom: 8 }}>AI Chatbot</h2>
      <ChatPage />
    </div>
  )
}
