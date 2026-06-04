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
  message: "Before we continue: ",
  question: "What do you think this explanation can tell you?",
  options: [
    "Which factors affected my result",
    "How the model behaves overall",
    "What actions I can take",
    "Increasing \"loans not paid off\" can improve my score"
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

// Dictionary to unify explanation labels for both the System Prompt and the UI Prompts
const EXPLANATION_DICT = {
  local: {
    system: "Local Feature Importance (SHAP Bar Plot)",
    ui: "Local Feature Importance"
  },
  counterfactual: {
    system: "Counterfactual explanation for the current applicant",
    ui: "Counterfactual Explanation"
  },
  global: {
    system: "Global Feature Importance (System-level SHAP Plot)",
    ui: "Global Feature Importance"
  }
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

  // Track the list of explanations the user wants to see in the dashboard
  const [selectedExplanations, setSelectedExplanations] = useState([
    "local",
  ])

  // Keep a ref of the selected explanations to safely access inside async callbacks
  const selectedExpsRef = useRef(selectedExplanations)
  useEffect(() => {
    selectedExpsRef.current = selectedExplanations
  }, [selectedExplanations])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, busy])

  // Dynamically build the text describing what's visible, reusing the mapping
  const visibleTexts = useMemo(() => {
    if (selectedExplanations.length === 0) {
      return {
        system: "No explanations currently visible",
        ui: "no explanations"
      }
    }
    return {
      system: selectedExplanations.map(k => EXPLANATION_DICT[k].system).join(", "),
      ui: selectedExplanations.map(k => EXPLANATION_DICT[k].ui).join(" and ")
    }
  }, [selectedExplanations])

  // The system prompt dynamically reads the current dashboard state.
  const system = useMemo(() => {
    return `You are a helpful assistant explaining a machine learning model used as an automated pre-qualification tool for credit line increase applications. The user represents applicant ID ${userInstanceId} in the dataset who are apply to increase their credit line. When answering questions, assume the user is asking about their own credit profile unless stated otherwise. The model produces a credit score from 0 to 100, where higher values indicate stronger chance for a credit line increase.

      Currently, the user has the following explanations visible on their dashboard:
      [ ${visibleTexts.system} ]
      If the user refers to "this explanation", "the chart", "the figure" or similar phrases, they are referring to these visible panels. Contextualise your answers based on what they can see.

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
      - Clearly distinguish between local explanations (single applicant) and global explanations (entire dataset or subgroup)
      - PROACTIVE TOOL CALLING: If the user asks whether they can infer certain information from the currently shown explanation(s), and the true answer requires a DIFFERENT explanation that is not currently shown (e.g., they ask about overall model behavior but only local importance is shown, or they ask for actionable changes but counterfactuals are missing), you MUST explain why the current explanation is insufficient and then IMMEDIATELY call the appropriate tool to generate and display the correct explanation in your response. Do not just tell them another explanation is needed.`
  }, [userInstanceId, visibleTexts.system])

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

        const getVizType = (toolName) => {
          if (toolName === "get_counterfactual_explanation") return "counterfactual"
          if (toolName === "generate_shap_bar_plot" || toolName === "generate_local_shap_bar_plot") return "local"
          if (toolName === "generate_shap_summary_plot") return "global"
          return "extra" // Unrecognized or extra charts
        }

        const counterfactuals = []
        const inlineVizes = []

        vizs.forEach(v => {
          const tool = v?.meta?.tool || v?.visualisation?.meta?.tool
          const vizType = getVizType(tool)

          // Extract counterfactual explicitly in case the dashboard needs to parse its target
          if (vizType === "counterfactual") {
            counterfactuals.push(v)
          }

          // If the explanation is NOT currently selected in the dashboard checklist, 
          // or it's an "extra" figure, display it inline in the chat message
          if (!selectedExpsRef.current.includes(vizType) || vizType === "extra") {
            inlineVizes.push(v)
          }
        })

        if (counterfactuals.length > 0) {
          setCounterfactualViz(JSON.parse(JSON.stringify(counterfactuals[0])))
        }

        if (inlineVizes.length > 0) {
          setMessages(prev => {
            const copy = [...prev]
            const last = copy[copy.length - 1]
            if (last?.role === "assistant") {
              last.visualisations = last.visualisations
                ? [...last.visualisations, ...inlineVizes]
                : [...inlineVizes]
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

  const handleToggleExplanation = (key) => {
    setSelectedExplanations(prev =>
      prev.includes(key) ? prev.filter(v => v !== key) : [...prev, key]
    )
  }

  return (
    <div style={{ display: "flex", gap: 12, height: "80vh", padding: 12, paddingTop: 30 }}>

      {/* Left side: Instance editor, Checklist, Dashboard */}
      <div
        style={{
          flex: 0.4,
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        {/* Instance Editor */}
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

        {/* Explanations Selection Panel */}
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            background: "#fafafa",
            padding: "12px 16px",
          }}
        >
          <div style={{ fontWeight: 600, fontSize: 13, color: "#475569", marginBottom: 8 }}>
            Explanations to display (for focus group activities)
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={selectedExplanations.includes("local")}
                onChange={() => handleToggleExplanation("local")}
              />
              Local Feature Importance
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={selectedExplanations.includes("counterfactual")}
                onChange={() => handleToggleExplanation("counterfactual")}
              />
              Counterfactual Explanation
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={selectedExplanations.includes("global")}
                onChange={() => handleToggleExplanation("global")}
              />
              Global Feature Importance
            </label>
          </div>
        </div>

        {/* Dashboard */}
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
            selectedExplanations={selectedExplanations}
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

        {/* Floating Design Switcher Buttons */}
        {showInitialSuggestions && (
          <div
            style={{
              position: "absolute",
              top: -42,
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
                        onClick={() => {
                          let promptText = `The explanation(s) currently shown on the dashboard: ${visibleTexts.ui}. The question is asking: "${DESIGN_B_CONTENT.question}". My answer is: "${opt}". Explain if I am correct or not. Also use the appropriate tool to generate and show which explanation can answer my question: "${opt}".`;
                          handleSend(promptText);
                        }}
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
                        <div
                          key={idx}
                          onClick={() => {
                            let promptText = `The explanation(s) currently shown on the dashboard: ${visibleTexts.ui}. I clicked the question: "${q}" under the category "This explanation CAN answer". Please explain why the currently shown explanation can answer this question.`;
                            handleSend(promptText);
                          }}
                          className="suggestion-btn"
                          style={{ background: "#eff6ff", flex: "1 1 auto" }}
                        >
                          {q}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Category 2: What it doesn't tell you */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 14 }}>
                      This explanation cannot answer:
                    </div>
                    {/* Display Questions Horizontally (Wrapping) */}
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {DESIGN_C_QUESTIONS.doesntTellYou.map((q, idx) => (
                        <div
                          key={idx}
                          onClick={() => {
                            let promptText = `The explanation(s) currently shown on the dashboard: ${visibleTexts.ui}. I clicked the question: "${q}" under the category "This explanation CANNOT answer". Please explain why the currently shown explanation cannot answer this question, and use the appropriate tool to generate and show the explanation that CAN answer it.`;
                            handleSend(promptText);
                          }}
                          className="suggestion-btn"
                          style={{ flex: "1 1 auto" }}
                        >
                          {q}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ fontSize: 13, color: "#6b7280", textAlign: "center" }}>
                    Click on a question to find out the explanation.
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