import React, { useMemo, useState, useEffect, useRef } from 'react'
import MessageList from './MessageList.jsx'
import MessageInput from './MessageInput.jsx'
import { chatOnce, chatWithToolsStream } from '../api.js'
import InstanceEditor from './InstanceEditor.jsx'
import Dashboard from './Dashboard.jsx'

// Which question would you like this explanation to help answer?
const DESIGN_A_QUESTIONS = [
  "Why is my score so low?",
  "What can I do to improve my score?",
  "What is the average score?",
]

// Loop: what else do you think the original explanation can answer?
const DESIGN_B_CONTENT = {
  message: "Before we continue: ",
  question: "What do you think this explanation can tell you?", // Select a question you think it can answer.
  options: {
    local: [
      "How each factor affected my score",
      "A higher \"on-time payment rate\" can increase my score",
      "Which factor is generally the most important",
      "What actions I can take to improve my score",
    ],
    counterfactual: [
      "How each factor affected my score",
      "A higher \"on-time payment rate\" can increase my score",
      "Which factor is generally the most important",
      "What actions I can take to improve my score",
    ],
    global: [
      "How each factor affected my score",
      "A higher \"on-time payment rate\" can increase my score",
      "Which factor is generally the most important",
      "What actions I can take to improve my score",
    ],
    cp: [
      "How each factor affected my score",
      "A higher \"on-time payment rate\" can increase my score",
      "Which factor is generally the most important",
      "What actions I can take to improve my score",
    ],
  }
}

// What would you like to explore next?
const DESIGN_C_QUESTIONS = {
  local: {
    tellsYou: [
      "What factors lowered/increased my score?",
      "How much did a factor affect my score?",
    ],
    doesntTellYou: [
      "What can I do to improve my score to 50?",
      "Which factors are generally important across all applicants?",
      "How would my score change if I have fewer loans?",
    ]
  },
  counterfactual: {
    tellsYou: [
      "What is the smallest change needed to get approved?",
    ],
    doesntTellYou: [
      "Why was my original score so low?",
      "Would decreasing my credit usage alone increase my score?",
      "Did my \"Loans not paid off\" negatively affect my score?"
    ]
  },
  global: {
    tellsYou: [
      "Which factors matter the most to the system in general?",
      "Does the system generally prioritise late payments or credit usage?",
    ],
    doesntTellYou: [
      "What changes should I make to increase my score?",
      "Why was my specific application denied?",
      "How much did \"credit used\" impact my score?",
    ]
  },
  cp: {
    tellsYou: [
      "How would my score change if I increased one factor?",
      "How sensitive is my score to each factor?"
    ],
    doesntTellYou: [
      "Why did I receive this score?",
      "Which factor affected my current score the most?",
      "What is the smallest change needed to get approved?",
    ]
  }
}

// Dictionary to unify explanation labels for both the System Prompt and the UI Prompts
const EXPLANATION_DICT = {
  local: {
    system: "Which factors pushed the applicant's score up or down (local feature importance)",
    ui: "What Affected Your Score",
    description: "how much each factor increased or decreased your score"
  },
  counterfactual: {
    system: "Smallest set of changes needed for the current applicant to reach target score",
    ui: "How to Improve Your Score",
    description: "the smallest change you could make to reach the target score"
  },
  cp: {
    system: "How one applicant's predicted credit score changes when changing a single factor",
    ui: "How Each Factor Affects Your Score",
    description: "how changing one factor at a time would affect your score"
  },
  global: {
    system: "Which factors matter the most across everyone",
    ui: "What Mattered Most Overall",
    description: "how important each factor is across all applicants"
  },
}

// Helper to dynamically generate the welcome message based on selected/clicked explanations
const generateWelcomeMessage = (explanations) => {
  if (explanations.length === 0) {
    return "The interface currently displays no explanations. Let me know if you have any questions.";
  }
  const descText = explanations
    .map(k => `**${EXPLANATION_DICT[k].ui}**, which shows ${EXPLANATION_DICT[k].description}`)
    .join(", and ");
  return `The interface currently displays ${descText}. Let me know if you have any questions.`;
};

export default function ChatPage() {
  const [userInstanceId] = useState(57)

  // Track the list of explanations the user wants to see in the dashboard
  const [selectedExplanations, setSelectedExplanations] = useState([
    "local",
  ])

  // NEW: Track whether the explanations menu is collapsed or expanded
  const [showExplanationsMenu, setShowExplanationsMenu] = useState(true)

  // Track which Design C and Design B questions have been clicked
  const [clickedQuestions, setClickedQuestions] = useState(new Set())

  // Initialise messages dynamically using the helper function
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: generateWelcomeMessage(["local"])
    }
  ])

  const [busy, setBusy] = useState(false)
  const [useStreaming, setUseStreaming] = useState(true)
  const [visualisations, setVisualisations] = useState([])
  const [counterfactualViz, setCounterfactualViz] = useState(null)
  const [cpVisualisations, setCpVisualisations] = useState([])
  const [backendHistory, setBackendHistory] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(true)
  const [llmStage, setLlmStage] = useState("thinking")
  const [activeDesign, setActiveDesign] = useState("A") // Toggles A, B, or C
  const messagesEndRef = useRef(null)

  // Keep a ref of the selected explanations to safely access inside async callbacks
  const selectedExpsRef = useRef(selectedExplanations)
  useEffect(() => {
    selectedExpsRef.current = selectedExplanations
  }, [selectedExplanations])

  // Update the initial message if the user clicks/toggles dashboard explanations BEFORE asking a question
  useEffect(() => {
    setMessages(prev => {
      // If the user has already sent a message, don't overwrite the chat history
      const hasUserMsg = prev.some(m => m.role === 'user');
      if (hasUserMsg) return prev;

      // Re-generate the message based on exactly what is clicked right now
      return [{ role: 'assistant', content: generateWelcomeMessage(selectedExplanations) }];
    });
  }, [selectedExplanations]);

  // Start a new log session on initial page load / refresh
  useEffect(() => {
    fetch("http://127.0.0.1:5001/api/session/reset")
      .then(res => res.json())
      .then(data => console.log("Initial session started:", data.log_file))
      .catch(err => console.error("Error starting session:", err));
  }, []);

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

  // Dynamically compile Design B options based on selected explanations
  const activeDesignBOptions = useMemo(() => {
    const options = new Set();
    selectedExplanations.forEach(exp => {
      if (DESIGN_B_CONTENT.options[exp]) {
        DESIGN_B_CONTENT.options[exp].forEach(opt => options.add(opt));
      }
    });
    return Array.from(options);
  }, [selectedExplanations]);

  // Dynamically compile Design C "tells you" questions
  const activeDesignCTellsYou = useMemo(() => {
    const questions = new Set();
    selectedExplanations.forEach(exp => {
      if (DESIGN_C_QUESTIONS[exp]?.tellsYou) {
        DESIGN_C_QUESTIONS[exp].tellsYou.forEach(q => questions.add(q));
      }
    });
    return Array.from(questions);
  }, [selectedExplanations]);

  // Dynamically compile Design C "doesn't tell you" questions
  const activeDesignCDoesntTellYou = useMemo(() => {
    const questions = new Set();
    selectedExplanations.forEach(exp => {
      if (DESIGN_C_QUESTIONS[exp]?.doesntTellYou) {
        DESIGN_C_QUESTIONS[exp].doesntTellYou.forEach(q => questions.add(q));
      }
    });
    return Array.from(questions);
  }, [selectedExplanations]);

  // The system prompt dynamically reads the current dashboard state.
  const system = useMemo(() => {
    return `You are a helpful assistant explaining a machine learning model used as an automated tool to approve or reject credit limit increase applications. The user represents applicant ID ${userInstanceId} in the dataset who are apply to increase their credit limit. When answering questions, assume the user is asking about their own credit profile unless stated otherwise. The model produces a credit score from 0 to 100, where higher values indicate stronger chance for a credit limit increase.

      Currently, the user has the following explanations visible on their dashboard:
      [ ${visibleTexts.system} ]
      If the user refers to "this explanation", "the chart", "the figure" or similar phrases, they are referring to these visible panels. Contextualise your answers based on what they can see. 
      IMPORTANT: Even though these explanations are displayed to the user, you do NOT automatically know what the actual data or results are. You MUST call the corresponding tool(s) to retrieve the data for these visible explanations so you can accurately understand the outputs and answer the user's questions. 

      Available factors/features include 6 variables:
      - Credit used (%) -- Percentage of available credit already used        
      - Months since last late payment -- How long since they last missed a payment
      - On-time payment rate (%) -- How often they've paid on time
      - Number of loans -- Number of loan accounts they've had
      - Loans not paid off (%) -- How many borrowing accounts still have debt on them
      - Months since last credit application -- How long since they last applied for credit

      Guidelines:
      - Keep answers concise, factual, and grounded in tool outputs.
      - Do NOT infer or assume missing values.
      - Do not guess or hallucinate the explanation results. Do not add your own interpretation!
      - If required inputs (e.g., instance_id, factor name, target) are missing, ask the user to provide them.
      - Clearly distinguish between advice for a single applicant versus trends for EVERYONE (global).
      - PROACTIVE TOOL CALLING: If the user asks whether they can infer certain information from the currently shown explanation(s), and the true answer requires a DIFFERENT explanation(s) that is not currently shown (e.g., they ask how to improve their score, but are looking at their current score breakdown), you MUST explain why the current explanation is insufficient and then IMMEDIATELY call the appropriate tool to generate the correct explanation in your response. Do not just tell them another explanation is needed.`
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
          if (toolName === "get_cp_plot" || toolName === "generate_all_cp_plots") return "cp"
          return "extra" // Unrecognised or extra charts
        }

        const counterfactuals = []
        const cpVizs = []
        const inlineVizes = []

        vizs.forEach(v => {
          const tool = v?.meta?.tool || v?.visualisation?.meta?.tool
          const vizType = getVizType(tool)

          // Extract specific types to pass down to Dashboard
          if (vizType === "counterfactual") {
            counterfactuals.push(v)
          }
          else if (vizType === "cp") {
            cpVizs.push(v)
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

        if (cpVizs.length > 0) {
          setCpVisualisations(cpVizs)
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

  const handleDesignCQuestion = (q, category) => {
    // Mark this exact question string as clicked
    setClickedQuestions(prev => new Set(prev).add(q))

    let promptText = ""
    if (category === "tellsYou") {
      promptText = `The explanation(s) currently shown on the dashboard: ${visibleTexts.ui}. I clicked the question: "${q}" under the category "This explanation CAN answer". Please explain why the currently shown explanation can answer this question, and also tell me the answer.`
    } else {
      promptText = `The explanation(s) currently shown on the dashboard: ${visibleTexts.ui}. I clicked the question: "${q}" under the category "This explanation CANNOT answer". Please explain why the currently shown explanation cannot answer this question, and use the appropriate tool to generate and show the explanation that CAN answer it.`
    }

    handleSend(promptText)
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

  // Calculate unclicked Design B options
  const unclickedDesignBOptions = activeDesignBOptions.filter(opt => !clickedQuestions.has(opt));

  return (
    <div style={{ display: "flex", gap: 12, height: "100%", padding: 12 }}>

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

        {/* Explanations Selection Panel (COLLAPSIBLE) */}
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            background: "#fafafa",
          }}
        >
          {/* Clickable Header */}
          <div
            onClick={() => setShowExplanationsMenu(!showExplanationsMenu)}
            style={{
              padding: "12px 16px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              cursor: "pointer",
              userSelect: "none"
            }}
          >
            <div style={{ fontWeight: 600, fontSize: 13, color: "#475569" }}>
              Explanations to display
            </div>

            {/* Chevron Icon */}
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#475569"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{
                transform: showExplanationsMenu ? "rotate(180deg)" : "rotate(0deg)",
                transition: "transform 0.2s ease"
              }}
            >
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>

          {/* Collapsible Content */}
          {showExplanationsMenu && (
            <div style={{
              padding: "0px 16px 12px 16px",
              display: "flex",
              flexDirection: "column",
              gap: 6
            }}>
              {Object.keys(EXPLANATION_DICT).map((key) => (
                <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={selectedExplanations.includes(key)}
                    onChange={() => handleToggleExplanation(key)}
                  />
                  {EXPLANATION_DICT[key].ui}
                </label>
              ))}
            </div>
          )}
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
            cpVisualisations={cpVisualisations}
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
            display: "flex",
            flexDirection: "column"
          }}
        >
          {/* Main Message History */}
          <MessageList messages={messages} busy={busy} status={llmStage} />

          {/* Persistent Box for Design C */}
          {activeDesign === "C" && !busy && (activeDesignCTellsYou.length > 0 || activeDesignCDoesntTellYou.length > 0) && (
            <div
              style={showInitialSuggestions ? {
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "80%",
                maxWidth: 450,
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
              } : {
                width: "80%",
                maxWidth: 450,
                margin: "20px auto 0 auto",
                background: "white",
                borderRadius: 16,
                boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
                border: "1px solid #e5e7eb",
                padding: 24,
                display: "flex",
                flexDirection: "column",
                gap: 12,
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

                {/* Category 1: What this tells you */}
                {activeDesignCTellsYou.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 14, color: "#2563eb" }}>
                      This explanation can answer:
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {activeDesignCTellsYou.map((q, idx) => {
                        const isClicked = clickedQuestions.has(q);
                        return (
                          <div
                            key={idx}
                            onClick={() => handleDesignCQuestion(q, "tellsYou")}
                            className="suggestion-btn"
                            style={{
                              display: "flex",
                              alignItems: "flex-start",
                              gap: "12px",
                              textAlign: "left",
                              padding: "8px 16px",
                              color: isClicked ? "#9ca3af" : undefined,
                              backgroundColor: isClicked ? "#f8fafc" : "#f3f4f6"
                            }}
                          >
                            <div style={{ marginTop: "2px", flexShrink: 0 }}>
                              {isClicked ? (
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="#2563eb">
                                  <circle cx="12" cy="12" r="12" />
                                  <path d="M7 12.5l3 3 7-7" stroke="#ffffff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                                </svg>
                              ) : (
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" strokeWidth="2.5">
                                  <circle cx="12" cy="12" r="10" />
                                </svg>
                              )}
                            </div>
                            <span style={{ lineHeight: "1.4" }}>{q}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Category 2: What it doesn't tell you */}
                {activeDesignCDoesntTellYou.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 14 }}>
                      This explanation cannot answer:
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {activeDesignCDoesntTellYou.map((q, idx) => {
                        const isClicked = clickedQuestions.has(q);
                        return (
                          <div
                            key={idx}
                            onClick={() => handleDesignCQuestion(q, "doesntTellYou")}
                            className="suggestion-btn"
                            style={{
                              display: "flex",
                              alignItems: "flex-start",
                              gap: "12px",
                              textAlign: "left",
                              padding: "8px 16px",
                              color: isClicked ? "#9ca3af" : undefined,
                              backgroundColor: isClicked ? "#f8fafc" : "#f3f4f6"
                            }}
                          >
                            <div style={{ marginTop: "2px", flexShrink: 0 }}>
                              {isClicked ? (
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="#2563eb">
                                  <circle cx="12" cy="12" r="12" />
                                  <path d="M7 12.5l3 3 7-7" stroke="#ffffff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                                </svg>
                              ) : (
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" strokeWidth="2.5">
                                  <circle cx="12" cy="12" r="10" />
                                </svg>
                              )}
                            </div>
                            <span style={{ lineHeight: "1.4" }}>{q}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                <div style={{ fontSize: 13, color: "#6b7280", textAlign: "center" }}>
                  Choose one to explore more.
                </div>
              </div>
            </div>
          )}

          {/* Persistent Box for Design B */}
          {activeDesign === "B" && !busy && unclickedDesignBOptions.length > 0 && (
            <div
              style={showInitialSuggestions ? {
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "80%",
                maxWidth: 450,
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
              } : {
                width: "80%",
                maxWidth: 450,
                margin: "20px auto 0 auto",
                background: "white",
                borderRadius: 16,
                boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
                border: "1px solid #e5e7eb",
                padding: 24,
                display: "flex",
                flexDirection: "column",
                gap: 12,
              }}
            >
              {/* Dynamic text based on whether it is the initial state or a subsequent turn */}
              {showInitialSuggestions ? (
                <>
                  <div style={{ fontWeight: 600, fontSize: 15, color: "#374151", textAlign: "center" }}>
                    {DESIGN_B_CONTENT.message}
                  </div>
                  <div style={{ fontSize: 14, color: "#4b5563", marginBottom: 8, textAlign: "center" }}>
                    {DESIGN_B_CONTENT.question}
                  </div>
                </>
              ) : (
                <div style={{ fontWeight: 600, fontSize: 15, color: "#374151", marginBottom: 4, textAlign: "center" }}>
                  What else do you think this explanation can tell you?
                </div>
              )}

              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {unclickedDesignBOptions.map((opt, idx) => (
                  <div
                    key={idx}
                    onClick={() => {
                      // Mark this option as clicked
                      setClickedQuestions(prev => new Set(prev).add(opt));

                      let promptText = `The explanation(s) currently shown on the dashboard: ${visibleTexts.ui}. The question is asking: "${showInitialSuggestions ? DESIGN_B_CONTENT.question : "What else do you think this explanation can tell you?"}". My answer is: "${opt}". Explain if I am correct or not. Also use the appropriate tool to generate and show which explanation can answer my question: "${opt}".`;
                      handleSend(promptText);
                    }}
                    className="suggestion-btn"
                  >
                    {opt}
                  </div>
                ))}
              </div>
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