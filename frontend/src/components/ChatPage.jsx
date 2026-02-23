import React, { useMemo, useState, useEffect, useRef } from 'react'
import MessageList from './MessageList.jsx'
import MessageInput from './MessageInput.jsx'
import { chatOnce, chatWithToolsStream } from '../api.js'
import VisualisationPanel from './VisualisationPanel.jsx'

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! Ask a question.' }
  ])
  const [busy, setBusy] = useState(false)
  const messagesEndRef = useRef(null)
  const [useStreaming, setUseStreaming] = useState(true)
  const [visualisations, setVisualisations] = useState([])
  const [backendHistory, setBackendHistory] = useState([])

  // const [useToolCalling, setUseToolCalling] = useState(false)
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });
  }, [messages, busy])
  
  const system = useMemo(
    () => 'You are a helpful assistant. Keep answers concise. Do not flatter. If a tool returned a missing argument, ask the user to provide it. Available features include MedInc (median income), AveBedrms (average number of bedrooms), AveRooms (average rooms), AveOccup (average number of occupants, HouseAge (house age), population, longitude and latitude.',
    []
  )

  // const historyForBackend = useMemo(() => {
  //   // send everything except the first assistant greeting if desired; keep it simple:
  //   return messages.filter(m => m.role !== 'assistant' || m.content !== 'Hi! Ask a question.')
  // }, [messages])

  async function handleSend(text) {
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])

    if (!useStreaming) {
      setBusy(true)
      try {
        const res = await chatOnce({
          message: text,
          history: backendHistory,  // TODO: update the history, typing bubble
          system,
          // model: 'llama3.2:3b'
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
    let assistantIndex = -1
    setMessages(prev => {
      assistantIndex = prev.length + 1
      return [...prev, { role: 'assistant', content: ''}]
    })
    setVisualisations([])

    // call the following function defined in api.js
    chatWithToolsStream({
      message: text,
      history: backendHistory,
      system,
      // model: 'llama3.2:3b',
      options: {
        temperature: 1
      },
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

  return (
    <div style={{ display: 'flex', gap: 12, height: '80vh', padding: 12 }}>
    {/* Left side - Visualisations */}
    <div style={{ flex: 0.4, border: '1px solid #ddd', borderRadius: 8, overflow: 'auto', background: '#fafafa' }}>
      <VisualisationPanel visualisations={visualisations} />
    </div>

    {/* Right side - Chat */}
    <div style={{ flex: 0.6, display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: 12, display: 'flex', gap: 12, alignItems: 'center' }}>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', background: '#f5f5f5', padding: '8px 12px', borderRadius: 6, fontSize: 12 }}>
          <input
            type="checkbox"
            checked={useStreaming}
            onChange={(e) => setUseStreaming(e.target.checked)}
          />
          Use streaming responses with tools
        </label>
        <span style={{ fontSize: 12, color: '#666' }}>
          {useStreaming ? 'Tools + Streaming' : 'Non-streaming'}
        </span>
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12, flex: 1, overflow: 'auto' }}>
        <MessageList messages={messages}  busy={busy} />
        <div ref={messagesEndRef} />
      </div>

      <div style={{ marginTop: 10 }}>
        <MessageInput disabled={busy} onSend={handleSend} />
      </div>
    </div>
  </div>
)
}
