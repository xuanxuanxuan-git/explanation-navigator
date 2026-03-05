import React, { useMemo, useState, useEffect, useRef } from 'react'
import MessageList from './MessageList.jsx'
import MessageInput from './MessageInput.jsx'
import { chatOnce, chatWithToolsStream } from '../api.js'
import VisualisationPanel from './VisualisationPanel.jsx'

const SUGGESTED_QUESTIONS = [
  "Why do I get this prediction?",
  "What is the most important feature for instance 2?",
  "What is the average model prediction?",
]

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! Ask a question.' }
  ])
  const [busy, setBusy] = useState(false)
  const [useStreaming, setUseStreaming] = useState(true)
  const [visualisations, setVisualisations] = useState([])
  const [backendHistory, setBackendHistory] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(true)
  const messagesEndRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, busy])

  const system = useMemo(
    () =>
      'You are a helpful assistant. Keep answers concise. Do not flatter. If a tool returned a missing argument, ask the user to provide it. Available features include MedInc (median income), AveBedrms (average number of bedrooms), AveRooms (average rooms), AveOccup (average number of occupants), HouseAge (house age), population, longitude and latitude.',
    []
  )

  async function handleSend(text) {
    if (!text?.trim()) return

    setShowSuggestions(false)

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])

    if (!useStreaming) {
      setBusy(true)
      try {
        const res = await chatOnce({
          message: text,
          history: backendHistory,
          system
        })
        setMessages(prev => [...prev, { role: 'assistant', content: res.reply }])
        setVisualisations([])
      } catch (e) {
        setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }])
      } finally {
        setBusy(false)
      }
      return
    }

    // streaming
    setBusy(true)
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])
    setVisualisations([])

    chatWithToolsStream({
      message: text,
      history: backendHistory,
      system,
      options: { temperature: 1 },
      onToken: (token) => {
        setMessages(prev => {
          const copy = [...prev]
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') {
            last.content += token
          }
          return copy
        })
      },
      onVisualisations: (vizs) => {
        setVisualisations(vizs || [])
      },
      // the complete message sent is {"done": true, "history": [...]}
      onDone: (payload) => {
        if (payload?.history) {
          setBackendHistory(payload.history)
        }
        setBusy(false)
      },
      onError: (err) => {
        setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${JSON.stringify(err)}` }])
        setBusy(false)
      }
    })
  }

  const showInitialSuggestions =
    showSuggestions &&
    !busy &&
    messages.filter(m => m.role === 'user').length === 0

  return (
    <div style={{ display: 'flex', gap: 12, height: '80vh', padding: 12 }}>
    {/* Left side: Visualisations */}
    <div style={{ flex: 0.4, border: '1px solid #ddd', borderRadius: 8, overflow: 'auto', background: '#fafafa' }}>
      <VisualisationPanel visualisations={visualisations} />
    </div>

      {/* Right side: Chat */}
      <div style={{ flex: 0.6, display: 'flex', flexDirection: 'column', position: 'relative' }}>

        {/* Chat Container */}
        <div style={{
          border: '1px solid #ddd',
          borderRadius: 8,
          padding: 12,
          flex: 1,
          overflow: 'auto',
          background: 'white'
        }}>
          <MessageList messages={messages} busy={busy} />
          <div ref={messagesEndRef} />
        </div>

        {/* Floating Dialogue Suggestion Box */}
        {showInitialSuggestions && (
          <div style={{
            position: 'absolute',
            bottom: 70,
            right: 20,
            width: 320,
            background: 'white',
            borderRadius: 16,
            boxShadow: '0 10px 30px rgba(0,0,0,0.12)',
            border: '1px solid #e5e7eb',
            padding: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            animation: 'fadeSlide 0.4s ease forwards',
            zIndex: 10
          }}>
            <div style={{
              fontWeight: 600,
              fontSize: 13,
              color: '#374151',
              marginBottom: 4
            }}>
              Try asking:
            </div>

            {SUGGESTED_QUESTIONS.map((q, idx) => (
              <div
                key={idx}
                onClick={() => handleSend(q)}
                style={{
                  padding: '8px 12px',
                  borderRadius: 8,
                  background: '#f3f4f6',
                  fontSize: 13,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={e => {
                  e.target.style.background = '#2563eb'
                  e.target.style.color = 'white'
                }}
                onMouseLeave={e => {
                  e.target.style.background = '#f3f4f6'
                  e.target.style.color = '#111'
                }}
              >
                {q}
              </div>
            ))}
          </div>
        )}

        {/* Input */}
        <div style={{ marginTop: 10 }}>
          <MessageInput disabled={busy} onSend={handleSend} />
        </div>
      </div>

      <style>
        {`
          @keyframes fadeSlide {
            from {
              opacity: 0;
              transform: translateY(15px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>

    </div>
  )
}