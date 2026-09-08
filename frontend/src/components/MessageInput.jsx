import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'

export default function MessageInput({ disabled, onSend }) {
  const [text, setText] = useState('')
  const inputRef = useRef(null)
  const MAX_HEIGHT = 280 // px 

  // Focus on initial mount and whenever the input becomes enabled again.
  // useEffect(() => {
  //   if (!disabled) inputRef.current?.focus()
  // }, [disabled])

  // Auto-resize on text changes
  useLayoutEffect(() => {
    const el = inputRef.current
    if (!el) return

    el.style.height = '0px' // reset so it can shrink too
    const next = Math.min(el.scrollHeight, MAX_HEIGHT)
    el.style.height = `${next}px`
    el.style.overflowY = el.scrollHeight > MAX_HEIGHT ? 'auto' : 'hidden'
    
    // Only auto-scroll if cursor is at the end
    if (el.selectionStart === text.length) {
      el.scrollTop = el.scrollHeight
    }
  }, [text])

  function submit(e) {
    e.preventDefault()
    const t = text.trim()
    if (!t || disabled) return

    // Clear text first 
    setText('')

    // Trigger parent send (which will disable input while waiting)
    onSend(t)

    // Optional: attempt focus immediately (useEffect will also refocus once enabled)
    // inputRef.current?.focus()
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' || e.key === 'Return') {
      if (e.shiftKey) {
        // Shift+Enter: insert newline at cursor
        e.preventDefault()
        const el = inputRef.current
        const start = el.selectionStart
        const end = el.selectionEnd
        const newText = text.slice(0, start) + '\n' + text.slice(end)
        setText(newText)
        // move cursor after the newline
        requestAnimationFrame(() => {
          el.selectionStart = el.selectionEnd = start + 1
        })
      } else {
        // Enter alone: submit
        e.preventDefault()
        submit(e)
      }
    }
  }

  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
      <textarea
        ref={inputRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
        rows={1}
        placeholder={disabled ? 'Waiting for answer...' : 'Ask a question'}
        style={{
          flex: 1,
          padding: '10px 12px',
          borderRadius: 8,
          border: '1px solid #ddd',
          resize: 'none',
          lineHeight: '20px',
          maxHeight: MAX_HEIGHT, // acts as a visual cap too
          overflowY: 'hidden',   // actual toggle happens in JS above
          boxSizing: 'border-box',
          fontFamily: 'inherit',
          fontSize: 14,
        }}
      />
  
      <button
        disabled={disabled}
        onClick={submit}
        aria-label="Send"
        style={{
          height: 40,
          width: 40,
          borderRadius: 8,         
          border: '1px solid #ddd',
          background: disabled ? '#e8eefc' : '#2563eb', // blue
          color: '#fff',
          display: 'grid',
          placeItems: 'center',
          fontSize: 18,
          lineHeight: 1,
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        ↑
      </button>
    </div>
  )
}
