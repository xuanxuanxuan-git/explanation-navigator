import React, { useMemo, useState, useEffect, useRef } from 'react'
import MessageList from './MessageList.jsx'
import MessageInput from './MessageInput.jsx'
import { chatOnce, chatWithToolsStream } from '../api.js'
import InstanceEditor from './InstanceEditor.jsx'
import Dashboard from './Dashboard.jsx'

const DESIGN_A_QUESTIONS = [
  "Why is my score so low?",
  "What can I do to improve my score?",
  "What is the average score?",
]

const DESIGN_B_CONTENT = {
  message: "Before we continue, let's test your understanding of the explanation.",
  question: "Based on the explanation, which feature has the highest impact on your credit score?",
  options: [
    "Credit used (%)",
    "Months since last late payment",
    "On-time payment rate (%)",
    "Number of loans"
  ]
}

const DESIGN_C_QUESTIONS = {
  tellsYou: [
    "What factors lowered my score?",
    "How much did a factor impact the decision?",
  ],
  doesntTellYou: [
    "What can I do to improve my score to 50?",
    "Does improving on-time payment rate improve my score?",
    "Does a factor affect my friend as much as it does on me?"
  ]
}

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
  const [activeDesign, setActiveDesign] = useState("A") // Toggles A, B, or C
  const messagesEndRef = useRef(null)
  const [userInstanceId] = useState(10)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, busy])

  const system = useMemo(
    () =>
      `You are a helpful assistant explaining a machine learning model used as an automated pre-qualification tool for credit line increase applications. The user represents applicant ID ${userInstanceId} in the dataset who are apply to increase their credit line. When answering questions, assume the user is asking about their own credit profile unless stated otherwise. The model produces a credit score from 0 to 100, where higher values indicate stronger chance for a credit line increase.

      Available features include 6 variables:
      - Credit used (%) -- Percentage of available credit already used        
      - Months since last late payment -- How long since they last missed a payment
      - On-time payment rate (%) -- How often they've paid on time
      - Number of loans -- Number of loan accounts they've had
      - Loans not paid off (%) -- How many borrowing accounts still have debt on them
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
          setMessages(prev => {
            const copy = [...prev]
            const last = copy[copy.length - 1]
            if (last?.role === "assistant") {
              last.visualisations = last.visualisations 
                ? [...last.visualisations, ...normalVizes] 
                : [...normalVizes]
            }
            return copy
          })
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
    // We add paddingTop: 50 here permanently so there is always room at the top for the buttons
    <div style={{ display: "flex", gap: 12, height: "80vh", padding: 12, paddingTop: 30 }}>
      
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
          position: "relative", // Ensures absolute children (like the design buttons) anchor to this container
        }}
      >
        
        {/* Floating Design Switcher Buttons (Centred to the chat side, floating above) */}
        {showInitialSuggestions && (
          <div 
            style={{ 
              position: "absolute", 
              top: -42, // Moves it into the 50px padding we created in the main container above
              left: "50%", 
              transform: "translateX(-50%)", 
              display: "flex", 
              gap: 8, 
              zIndex: 20 
            }}
          >
            {["A", "B", "C"].map((design) => (
              <button
                key={design}
                onClick={() => setActiveDesign(design)}
                style={{
                  padding: "6px 16px",
                  borderRadius: 20,
                  border: "1px solid #cbd5e1",
                  background: activeDesign === design ? "#2563eb" : "#f8fafc",
                  color: activeDesign === design ? "white" : "#334155",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 600,
                  transition: "all 0.2s ease"
                }}
              >
                Design {design}
              </button>
            ))}
          </div>
        )}

        {/* Chat Container */}
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            padding: 12,
            flex: 1,
            overflow: "auto",
            background: "white",
            position: "relative",
          }}
        >
          <MessageList messages={messages} busy={busy} status={llmStage} />
          
          {/* Centred Floating Dialogue Suggestion Box */}
          {showInitialSuggestions && (
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "80%",
                maxWidth: 500,
                background: "white",
                borderRadius: 16,
                boxShadow: "0 10px 30px rgba(0,0,0,0.12)",
                border: "1px solid #e5e7eb",
                padding: 24,
                display: "flex",
                flexDirection: "column",
                gap: 12,
                animation: "fadeSlide 0.4s ease forwards",
                zIndex: 10,
              }}
            >
              {/* DESIGN A */}
              {activeDesign === "A" && (
                <>
                  <div style={{ fontWeight: 600, fontSize: 15, color: "#374151", marginBottom: 4, textAlign: "center" }}>
                    You can ask:
                  </div>
                  {DESIGN_A_QUESTIONS.map((q, idx) => (
                    <div
                      key={idx}
                      onClick={() => handleSend(q)}
                      className="suggestion-btn"
                    >
                      {q}
                    </div>
                  ))}
                </>
              )}

              {/* DESIGN B */}
              {activeDesign === "B" && (
                <>
                  <div style={{ fontWeight: 600, fontSize: 15, color: "#374151", textAlign: "center" }}>
                    {DESIGN_B_CONTENT.message}
                  </div>
                  <div style={{ fontSize: 14, color: "#4b5563", marginBottom: 8, textAlign: "center" }}>
                    {DESIGN_B_CONTENT.question}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {DESIGN_B_CONTENT.options.map((opt, idx) => (
                      <div
                        key={idx}
                        onClick={() => handleSend(`The question is asking: ${DESIGN_B_CONTENT.question}. My answer is: ${opt}. Explain if I am correct or not.`)}
                        className="suggestion-btn"
                      >
                        {opt}
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* DESIGN C */}
              {activeDesign === "C" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                  
                  {/* Category 1: What this tells you */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 14, color: "#2563eb" }}>
                    This explanation can answer:
                    </div>
                    {/* Display Questions Horizontally (Wrapping) */}
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {DESIGN_C_QUESTIONS.tellsYou.map((q, idx) => (
                        <div key={idx} onClick={() => handleSend(q)} className="suggestion-btn" style={{ background: "#eff6ff",flex: "1 1 auto" }}>
                          {q}
                        </div>
                      ))}
                    </div>
                  </div>

                 {/* Category 2: What it doesn't tell you */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 14}}>
                      This explanation cannot answer:
                    </div>
                    {/* Display Questions Horizontally (Wrapping) */}
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {DESIGN_C_QUESTIONS.doesntTellYou.map((q, idx) => (
                        <div key={idx} onClick={() => handleSend(q)} className="suggestion-btn" style={{  flex: "1 1 auto" }}>
                          {q}
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              )}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div style={{ marginTop: 10 }}>
          <MessageInput disabled={busy} onSend={handleSend} />
        </div>

        <style>
          {`
          @keyframes fadeSlide {
            from {
              opacity: 0;
              transform: translate(-50%, -40%);
            }
            to {
              opacity: 1;
              transform: translate(-50%, -50%);
            }
          }

          .suggestion-btn {
            padding: 10px 14px;
            border-radius: 8px;
            background: #f3f4f6;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
          }

          .suggestion-btn:hover {
            background: #2563eb !important;
            color: white !important;
          }
        `}
        </style>
      </div>
    </div>
  )
}