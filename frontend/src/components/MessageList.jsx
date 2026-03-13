import React from 'react'
import TypingStatus from './TypingStatus.jsx'

export default function MessageList({ messages, busy, status }) {

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {messages.map((m, idx) => (
        <div
          key={idx}
          style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '80%',
            padding: '10px 12px',
            borderRadius: 12,
            background: m.role === 'user' ? '#e8f0fe' : '#f5f5f5',
            whiteSpace: 'pre-wrap'
          }}
        >
          <div style={{ fontSize: 12, opacity: 0.65, marginBottom: 4 }}>
            {m.role}
          </div>
          <div>
            {m.role === 'assistant' && !m.content && busy ? (
              <TypingStatus status={status}/>
            ) : (
              m.content
            )}
          </div>
        </div>
      ))}
    </div>
  )
}