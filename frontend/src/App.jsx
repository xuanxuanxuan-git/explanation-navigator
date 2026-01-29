import React, { useState } from 'react'
import ChatPage from './components/ChatPage.jsx'
import ChatWidget from './components/ChatWidget.jsx'

export default function App() {
  const [mode, setMode] = useState('page') // 'page' or 'widget'

  return (
    <div style={{ fontFamily: 'system-ui, Arial', padding: 16 }}>
      <h2>LLM Chatbot</h2>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button onClick={() => setMode('page')}>Full page</button>
        <button onClick={() => setMode('widget')}>Widget embed demo</button>
      </div>

      {mode === 'page' ? <ChatPage /> : <ChatWidget />}
    </div>
  )
}
