// Central place to talk to your Flask backend

const BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001'

export async function health() {
  const r = await fetch(`${BASE_URL}/api/health`)
  return r.json()
}

export async function chatOnce({ message, history = [], model, system, options }) {
  const r = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, model, system, options })
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

// add function about api/chat/tools

// SSE streaming
export function chatStream({ message, history = [], model, system, options, onToken, onDone, onError }) {
  fetch(`${BASE_URL}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, model, system, options })
  }).then(async (res) => {
    if (!res.ok) throw new Error(await res.text())
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // Parse SSE frames separated by blank line
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''

      for (const p of parts) {
        const lines = p.split('\n')
        let event = 'message'
        let data = ''
        for (const line of lines) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          if (line.startsWith('data:')) data += line.slice(5).trim()
        }

        try {
          const obj = data ? JSON.parse(data) : {}
          if (event === 'token') onToken?.(obj.token || '', obj.done)
          if (event === 'done') onDone?.()
          if (event === 'error') onError?.(obj)
        } catch (e) {
          // ignore parse errors
        }
      }
    }
  }).catch((err) => onError?.({ message: err.message }))
}

// Tool calling with streaming final response
export function chatWithToolsStream({ message, history = [], model, system, options, onToken, onDone, onVisualisations, onError }) {
  fetch(`${BASE_URL}/api/chat/tools/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, model, system, options })
  }).then(async (res) => {
    if (!res.ok) throw new Error(await res.text())
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''

      for (const p of parts) {
        const lines = p.split('\n')
        let event = 'message'
        let data = ''
        for (const line of lines) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          if (line.startsWith('data:')) data += line.slice(5).trim()
        }

        try {
          const obj = data ? JSON.parse(data) : {}
          if (event === 'token') onToken?.(obj.token || '', obj.done)
          if (event === 'visualisations') onVisualisations?.(obj.visualisations || [])
          if (event === 'done') onDone?.(obj)
          if (event === 'error') onError?.(obj)
        } catch (e) {
          // ignore parse errors
        }
      }
    }
  }).catch((err) => onError?.({ message: err.message }))
}

export async function fetchInstance(instanceId) {
  const res = await fetch(`${BASE_URL}/api/instance/${instanceId}`)
  if (!res.ok) {
    throw new Error(await res.text())
  }
  return res.json()
}

export async function predictInstanceWithChanges({ instance_id, changes }) {
  const res = await fetch(`${BASE_URL}/api/instance/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ instance_id, changes }),
  })
  if (!res.ok) {
    throw new Error(await res.text())
  }
  return res.json()
}

export async function generateShapBarPlot(instanceId, maxDisplay = 10) {
  const res = await fetch(
    `${BASE_URL}/api/instance/${instanceId}/shap-bar-plot?max_display=${encodeURIComponent(
      maxDisplay
    )}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  )
  if (!res.ok) {
    throw new Error(await res.text())
  }
  return res.json()
}

export async function generateCounterfactualExplanation(
  instanceId,
  target = 0.5,
  maxSteps = 50,
) {
  const params = new URLSearchParams({
    target: target.toString(),
    max_steps: maxSteps.toString(),
  })
  const res = await fetch(
    `${BASE_URL}/api/instance/${instanceId}/counterfactual-explanation?${params}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    },
  )
  if (!res.ok) {
    throw new Error(await res.text())
  }
  return res.json()
}