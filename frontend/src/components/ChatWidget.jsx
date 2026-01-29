import React, { useMemo, useState, useEffect, useRef } from 'react'
import MessageList from './MessageList.jsx'
import MessageInput from './MessageInput.jsx'
import { chatStream } from '../api.js'
import './widget.css'

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [messages, setMessages] = useState([
    { role: 'Assistant', content: 'Hi! How can I help?' }
  ])
  const messagesEndRef = useRef(null)
  
  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });
  }, [messages])

  const system = useMemo(
    () => 'You are a website support assistant. Ask clarifying questions when needed.',
    []
  )

  function handleSend(text) {
    setMessages(prev => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    setBusy(true)

    const history = messages

    chatStream({
      message: text,
      history,
      system,
      // model: 'llama3.2:3b',
      onToken: (token) => {
        setMessages(prev => {
          const copy = [...prev]
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') last.content += token
          return copy
        })
      },
      onDone: () => setBusy(false),
      onError: (err) => {
        setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${JSON.stringify(err)}` }])
        setBusy(false)
      }
    })
  }

  return (
    <>
      <button className="chat-fab" onClick={() => setOpen(o => !o)}>
        Chat
      </button>

      {open && (
        <div className="chat-panel">
          <div className="chat-panel-header">
            <div>Support</div>
            <button onClick={() => setOpen(false)} style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}>
              Close
            </button>
          </div>

          <div className="chat-panel-body">
            <MessageList messages={messages} />
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-panel-footer">
            <MessageInput disabled={busy} onSend={handleSend} />
          </div>
        </div>
      )}
    </>
  )
}
