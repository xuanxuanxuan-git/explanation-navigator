import React, { useMemo, useState, useEffect, useRef } from 'react'
import MessageList from './MessageList.jsx'
import MessageInput from './MessageInput.jsx'
import { chatOnce, chatWithToolsStream } from '../api.js'
import InstanceEditor from './InstanceEditor.jsx'
import Dashboard from './Dashboard.jsx'

const SUGGESTED_QUESTIONS = [
  "Why is my risk of default high?",
  "What can I change to reduce my risk?",
  "What is the average probability of default?",
]

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! Ask a question.' }
  ])
  const [busy, setBusy] = useState(false)
  const [useStreaming, setUseStreaming] = useState(true)
  const [visualisations, setVisualisations] = useState([])
  const [counterfactualViz, setCounterfactualViz] = useState(null)
  const [backendHistory, setBackendHistory] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(true)
  const [llmStage, setLlmStage] = useState("thinking")
  const messagesEndRef = useRef(null)
  const [userInstanceId] = useState(3)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, busy])

  const system = useMemo(
    () =>
      `You are a helpful assistant explaining a machine learning model for credit risk prediction. The user represents applicant ID ${userInstanceId} in the dataset. When answering questions, assume the user is asking about their own credit profile unless stated otherwise. The model predicts probability of default (credit risk), where higher values indicate higher likelihood of default. 

      Available features include 6 variables:
      - Credit used (%) -- Percentage of available credit already used        
      - Months since last late payment -- How long since they last missed a payment
      - On-time payment rate (%) -- How often they've paid on time
      - Total credit trades -- Number of accounts they've had
      - Trades with unpaid balance (%) -- How many borrowing accounts still have debt on them
      - Months since last credit application -- How long since they last applied for credit

      Guidelines:
      - Keep answers concise, factual, and grounded in tool outputs
      - Do NOT infer or assume missing values
      - Do NOT hallucinate feature values or explanations
      - If required inputs (e.g., instance_id, feature, target) are missing, ask the user to provide them
      - Clearly distinguish between local explanations (single applicant) and global explanations (entire dataset or subgroup)`,
    [userInstanceId],
  )

  async function handleSend(text) {
    if (!text?.trim()) return

    setShowSuggestions(false)

    const userMsg = { role: "user", content: text }
    setMessages(prev => [...prev, userMsg])

    if (!useStreaming) {
      setBusy(true)
      try {
        const res = await chatOnce({
          message: text,
          history: backendHistory,
          system,
        })
        setMessages(prev => [
          ...prev,
          { role: "assistant", content: res.reply },
        ])
        setVisualisations([])
      } catch (e) {
        setMessages(prev => [
          ...prev,
          { role: "assistant", content: `Error: ${e.message}` },
        ])
      } finally {
        setBusy(false)
      }
      return
    }

    // streaming
    setBusy(true)
    setLlmStage("thinking")
    setMessages(prev => [...prev, { role: "assistant", content: "" }])

    chatWithToolsStream({
      message: text,
      history: backendHistory,
      system,
      options: { temperature: 1 },
      onToken: token => {
        setMessages(prev => {
          const copy = [...prev]
          const last = copy[copy.length - 1]
          if (last?.role === "assistant") {
            last.content += token
          }
          return copy
        })
      },
      onVisualisations: vizs => {
        if (!vizs?.length) return
      
        const counterfactuals = vizs.filter(
          v =>
            v?.meta?.tool === "get_counterfactual_explanation" ||
            v?.visualisation?.meta?.tool ===
              "get_counterfactual_explanation"
        )
      
        const normalVizes = vizs.filter(v => {
          const tool =
            v?.meta?.tool ||
            v?.visualisation?.meta?.tool
        
          return (
            tool !== "get_counterfactual_explanation" &&
            tool !== "generate_shap_bar_plot"
          )
        })
      
        if (counterfactuals.length > 0) {
          setCounterfactualViz(
            JSON.parse(JSON.stringify(counterfactuals[0]))
          )
        }
      
        if (normalVizes.length > 0) {
          setVisualisations(prev => [...prev, ...normalVizes])
        }
      },
      onDone: payload => {
        if (payload?.history) {
          setBackendHistory(payload.history)
        }
        setBusy(false)
      },
      onError: err => {
        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${JSON.stringify(err)}`,
          },
        ])
        setBusy(false)
      },
    })
  }

  const showInitialSuggestions =
    showSuggestions &&
    !busy &&
    messages.filter(m => m.role === "user").length === 0

  return (
    <div style={{ display: "flex", gap: 12, height: "80vh", padding: 12 }}>
      {/* Left side: Instance editor + dashboard */}
      <div
        style={{
          flex: 0.4,
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            background: "#fafafa",
            overflow: "auto",
          }}
        >
          <InstanceEditor instanceId={userInstanceId} />
        </div>

        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            background: "#fafafa",
            overflow: "auto",
            flex: 1,
            minHeight: 0,
          }}
        >
          <Dashboard
            instanceId={userInstanceId}
            visualisations={visualisations}
            counterfactualViz={counterfactualViz}
          />
        </div>
      </div>

      {/* Right side: Chat */}
      <div
        style={{
          flex: 0.6,
          display: "flex",
          flexDirection: "column",
          position: "relative",
        }}
      >
        {/* Chat Container */}
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            padding: 12,
            flex: 1,
            overflow: "auto",
            background: "white",
          }}
        >
          <MessageList messages={messages} busy={busy} status={llmStage} />
          <div ref={messagesEndRef} />
        </div>

        {/* Floating Dialogue Suggestion Box */}
        {showInitialSuggestions && (
          <div
            style={{
              position: "absolute",
              bottom: 70,
              right: 20,
              width: 320,
              background: "white",
              borderRadius: 16,
              boxShadow: "0 10px 30px rgba(0,0,0,0.12)",
              border: "1px solid #e5e7eb",
              padding: 16,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              animation: "fadeSlide 0.4s ease forwards",
              zIndex: 10,
            }}
          >
            <div
              style={{
                fontWeight: 600,
                fontSize: 13,
                color: "#374151",
                marginBottom: 4,
              }}
            >
              Try asking:
            </div>

            {SUGGESTED_QUESTIONS.map((q, idx) => (
              <div
                key={idx}
                onClick={() => handleSend(q)}
                style={{
                  padding: "8px 12px",
                  borderRadius: 8,
                  background: "#f3f4f6",
                  fontSize: 13,
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={e => {
                  e.target.style.background = "#2563eb"
                  e.target.style.color = "white"
                }}
                onMouseLeave={e => {
                  e.target.style.background = "#f3f4f6"
                  e.target.style.color = "#111"
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
    </div>
  )
}