import React, { useEffect, useMemo, useRef, useState } from "react"
import Plot from "react-plotly.js"
import {
  generateShapBarPlot,
  generateCounterfactualExplanation,
  generateGlobalShapPlot,
  generateAllCpPlots
} from "../api.js"

/* ---------- HELPERS ---------- */

function normaliseVisualisation(viz, fallbackTitle = "Visualisation") {
  if (!viz) return null

  if (viz.type === "plotly") {
    return {
      type: "plotly",
      title: viz.title || viz.figure?.layout?.title?.text || fallbackTitle,
      figure: {
        data: viz.figure?.data || [],
        layout: viz.figure?.layout || {},
      },
      config: viz.config || { responsive: true },
      meta: viz.meta || {},
    }
  }

  if (viz.visualisation?.type === "plotly") {
    return {
      type: "plotly",
      title:
        viz.visualisation?.title ||
        viz.visualisation?.figure?.layout?.title?.text ||
        fallbackTitle,
      figure: {
        data: viz.visualisation?.figure?.data || [],
        layout: viz.visualisation?.figure?.layout || {},
      },
      config: viz.visualisation?.config || { responsive: true },
      meta: viz.visualisation?.meta || {},
      target: viz.target || viz.visualisation?.meta?.target,
    }
  }

  return null
}

/* ---------- MAIN COMPONENT ---------- */

export default function Dashboard({
  instanceId,
  visualisations = [],
  counterfactualViz,
  cpVisualisations = [], // Data from chat stream (if LLM proactively calls it)
  selectedExplanations = []
}) {
  const showLocal = selectedExplanations.includes("local");
  const showCF = selectedExplanations.includes("counterfactual");
  const showGlobal = selectedExplanations.includes("global");
  const showCP = selectedExplanations.includes("cp");

  /* SHAP state */
  const [shapViz, setShapViz] = useState(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState("")

  /* Counterfactual state */
  const [cfViz, setCfViz] = useState(null)
  const [cfLoading, setCfLoading] = useState(false)
  const [cfError, setCfError] = useState("")
  const [target, setTarget] = useState(50)
  const [cfUpdating, setCfUpdating] = useState(false)

  /* Global state */
  const [globalViz, setGlobalViz] = useState(null)
  const [globalLoading, setGlobalLoading] = useState(false)
  const [globalError, setGlobalError] = useState("")

  /* CP Plots state */
  const [cpViz, setCpViz] = useState(null)
  const [cpLoading, setCpLoading] = useState(false)
  const [cpError, setCpError] = useState("")

  /* Extra figures carousel state */
  const [vizIndex, setVizIndex] = useState(0)
  const prevExtraCountRef = useRef(0)

  /* ---------- LOAD GLOBAL ---------- */
  useEffect(() => {
    if (!showGlobal) return
    let cancelled = false;

    async function loadGlobal() {
      setGlobalLoading(true)
      setGlobalError("")
      try {
        const res = await generateGlobalShapPlot()
        if (cancelled) return
        const norm = normaliseVisualisation(res, `Global Feature Importance`)
        setGlobalViz(norm)
      } catch (e) {
        if (cancelled) return
        setGlobalError(e.message || "Failed to load Global visualisation")
        setGlobalViz(null)
      } finally {
        if (!cancelled) setGlobalLoading(false)
      }
    }

    if (!globalViz && !globalLoading) {
      loadGlobal()
    }

    return () => { cancelled = true }
  }, [showGlobal, globalViz])


  /* ---------- LOAD SHAP PLOT ---------- */
  useEffect(() => {
    if (!showLocal) return

    let cancelled = false;
    async function loadShap() {
      setShapLoading(true)
      setShapError("")
      try {
        const res = await generateShapBarPlot(instanceId)
        if (cancelled) return
        const norm = normaliseVisualisation(
          res,
          `Local feature attribution for applicant ${instanceId}`,
        )
        setShapViz(norm)
      } catch (e) {
        if (cancelled) return
        setShapError(e.message || "Failed to load SHAP visualisation")
        setShapViz(null)
      } finally {
        if (!cancelled) setShapLoading(false)
      }
    }

    loadShap()
    return () => { cancelled = true }
  }, [instanceId, showLocal])

  /* ---------- LOAD COUNTERFACTUAL ---------- */
  useEffect(() => {
    if (!counterfactualViz) return

    const norm = normaliseVisualisation(
      counterfactualViz,
      `Counterfactual explanation for applicant ${instanceId}`
    )

    setCfViz(norm)
    setCfLoading(false)
    setCfError("")

    const t =
      counterfactualViz?.meta?.target ??
      counterfactualViz?.target ??
      counterfactualViz?.data?.target ??
      counterfactualViz?.visualisation?.meta?.target

    const parsed = Number(t)

    if (!Number.isNaN(parsed)) {
      setTarget(parsed)
    }
  }, [counterfactualViz, instanceId])

  const loadCounterfactual = async (tgt = target) => {
    setCfLoading(true)
    setCfError("")
    try {
      const res = await generateCounterfactualExplanation(instanceId, tgt)
      const norm = normaliseVisualisation(
        res,
        `Counterfactual explanation for applicant ${instanceId}`,
      )
      setCfViz(norm)
      const returnedTarget =
        res?.data?.target ??
        res?.target ??
        res?.meta?.target ??
        res?.visualisation?.meta?.target

      if (returnedTarget !== undefined && returnedTarget !== null) {
        const parsed =
          typeof returnedTarget === "string"
            ? parseFloat(returnedTarget)
            : Number(returnedTarget)

        if (!Number.isNaN(parsed)) {
          setTarget(parsed)
        }
      }
    } catch (e) {
      setCfError(e.message || "Failed to load counterfactual")
      setCfViz(null)
    } finally {
      setCfLoading(false)
    }
  }

  useEffect(() => {
    if (!showCF) return
    loadCounterfactual()
  }, [instanceId, showCF])

  const handleTargetChange = e => {
    const newTarget = e.target.value === "" ? "" : parseFloat(e.target.value)
    setTarget(newTarget)
  }

  const handleUpdateTarget = async () => {
    if (cfUpdating) return
    setCfUpdating(true)
    await loadCounterfactual(target)
    setCfUpdating(false)
  }

  /* ---------- LOAD CP PLOTS ---------- */
  
  // 1) Set state from proactive LLM generations if it matches
  useEffect(() => {
    if (cpVisualisations && cpVisualisations.length > 0) {
      const norm = normaliseVisualisation(
        cpVisualisations[0],
        `Ceteris Paribus for applicant ${instanceId}`
      )
      setCpViz(norm)
      setCpLoading(false)
      setCpError("")
    }
  }, [cpVisualisations, instanceId])

  // 2) Or auto-load it based on the dashboard checkbox
  useEffect(() => {
    // If we don't want to show it, do nothing
    if (!showCP) return
    
    // If the LLM just proactively generated it, don't immediately overwrite it
    if (cpViz) return 

    let cancelled = false;
    async function loadCP() {
      setCpLoading(true)
      setCpError("")
      try {
        const res = await generateAllCpPlots(instanceId)
        if (cancelled) return
        const norm = normaliseVisualisation(res, `Ceteris Paribus for applicant ${instanceId}`)
        setCpViz(norm)
      } catch (e) {
        if (cancelled) return
        setCpError(e.message || "Failed to load CP visualisation")
        setCpViz(null)
      } finally {
        if (!cancelled) setCpLoading(false)
      }
    }

    loadCP()

    return () => { cancelled = true }
  }, [instanceId, showCP]) // Removed cpViz and cpLoading from here!


  /* ---------- EXTRA FIGURES ---------- */

  const extraFigures = useMemo(
    () =>
      (visualisations || [])
        .map((viz, idx) =>
          normaliseVisualisation(viz, `Visualisation ${idx + 1}`),
        )
        .filter(Boolean)
        .map((v, i) => ({ ...v, key: `extra-${i}` })),
    [visualisations],
  )

  const hasExtra = extraFigures.length > 0
  const currentExtra = hasExtra ? extraFigures[vizIndex] : null

  useEffect(() => {
    const count = extraFigures.length
    const prevCount = prevExtraCountRef.current

    if (count === 0) {
      setVizIndex(0)
    } else if (count > prevCount) {
      setVizIndex(count - 1)
    } else {
      setVizIndex(i => Math.min(i, Math.max(0, count - 1)))
    }

    prevExtraCountRef.current = count
  }, [extraFigures.length])

  function prevViz() {
    setVizIndex(i => (i > 0 ? i - 1 : i))
  }

  function nextViz() {
    setVizIndex(i =>
      i < extraFigures.length - 1 ? i + 1 : i,
    )
  }

  /* ---------- RENDER ---------- */

  // Provide an empty state if no explanations are selected
  if (!showLocal && !showCF && !showGlobal && !showCP && !hasExtra) {
    return (
      <div style={{ padding: 20, color: "#6b7280", textAlign: "center", fontSize: 13 }}>
        No explanations selected. Please check options on the left.
      </div>
    );
  }

  return (
    <div
      style={{
        padding: 10,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        height: "100%",
        boxSizing: "border-box",
        overflowY: "auto",
      }}
    >

      {/* ===== SHAP SECTION (Local) ===== */}
      {showLocal && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {shapLoading && !shapViz && (
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Loading Local SHAP explanation
            </div>
          )}

          {shapError && (
            <div style={{ padding: 8, borderRadius: 8, background: "#fee2e2", color: "#991b1b", fontSize: 12 }}>
              {shapError}
            </div>
          )}

          {shapViz?.type === "plotly" && (
            <div style={{ borderRadius: 10, overflow: "hidden", border: "1px solid #e5e7eb", background: "#fff" }}>
              <Plot
                data={shapViz.figure?.data || []}
                layout={{
                  autosize: true,
                  height: 260,
                  margin: { l: 140, r: 20, t: 55, b: 40 },
                  ...(shapViz.figure?.layout || {}),
                }}
                config={{
                  responsive: true,
                  displaylogo: false,
                  ...(shapViz.config || {}),
                }}
                style={{ width: "100%", height: 260 }}
                useResizeHandler
              />
            </div>
          )}
        </div>
      )}

      {/* ===== COUNTERFACTUAL SECTION ===== */}
      {showCF && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", overflow: "hidden" }}>
            {/* Target input bar */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 16px", background: "#f8fafc", borderBottom: "1px solid #e5e7eb" }}>
              <label style={{ fontSize: 12, color: "#64748b", minWidth: 50 }}>
                Target:
              </label>
              <input
                type="number"
                step="1"
                min="0"
                max="100"
                value={target}
                onChange={handleTargetChange}
                style={{ flex: 1, padding: "6px 8px", border: "1px solid #cbd5e1", borderRadius: 6, fontSize: 13 }}
              />
              <button
                onClick={handleUpdateTarget}
                disabled={cfUpdating}
                style={{
                  padding: "6px 16px",
                  borderRadius: 6,
                  border: "1px solid #3b82f6",
                  background: "#3b82f6",
                  color: "white",
                  fontSize: 12,
                  fontWeight: 500,
                  cursor: cfUpdating ? "default" : "pointer",
                  opacity: cfUpdating ? 0.7 : 1,
                }}
              >
                {cfUpdating ? "Updating..." : "Update"}
              </button>
            </div>

            {/* Plot below target input */}
            <div style={{ padding: "12px 0px" }}>
              {cfLoading && !cfViz && (
                <div style={{ fontSize: 12, color: "#6b7280" }}>
                  Loading counterfactual
                </div>
              )}

              {cfError && (
                <div style={{ padding: 8, borderRadius: 8, background: "#fee2e2", color: "#991b1b", fontSize: 12 }}>
                  {cfError}
                </div>
              )}

              {cfViz?.type === "plotly" && (
                <div style={{ overflow: "hidden", background: "#fff" }}>
                  <Plot
                    data={cfViz.figure?.data || []}
                    layout={{
                      autosize: true,
                      height: 300,
                      ...(cfViz.figure?.layout || {}),
                    }}
                    config={{
                      responsive: true,
                      displaylogo: false,
                      ...(cfViz.config || {}),
                    }}
                    style={{ width: "100%", height: "100%" }}
                    useResizeHandler
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ===== GLOBAL SECTION ===== */}
      {showGlobal && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {globalLoading && !globalViz && (
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Loading Global explanation
            </div>
          )}

          {globalError && (
            <div style={{ padding: 8, borderRadius: 8, background: "#fee2e2", color: "#991b1b", fontSize: 12 }}>
              {globalError}
            </div>
          )}

          {globalViz?.type === "plotly" && (
            <div style={{ borderRadius: 10, overflow: "hidden", border: "1px solid #e5e7eb", background: "#fff" }}>
              <Plot
                data={globalViz.figure?.data || []}
                layout={{
                  autosize: true,
                  height: 260,
                  margin: { l: 140, r: 20, t: 55, b: 40 },
                  ...(globalViz.figure?.layout || {}),
                }}
                config={{
                  responsive: true,
                  displaylogo: false,
                  ...(globalViz.config || {}),
                }}
                style={{ width: "100%", height: 260 }}
                useResizeHandler
              />
            </div>
          )}
        </div>
      )}

      {/* ===== CETERIS PARIBUS (CP) SECTION ===== */}
      {showCP && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {cpLoading && !cpViz && (
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Loading CP Plots
            </div>
          )}

          {cpError && (
            <div style={{ padding: 8, borderRadius: 8, background: "#fee2e2", color: "#991b1b", fontSize: 12 }}>
              {cpError}
            </div>
          )}

          {cpViz?.type === "plotly" && (
            <div style={{ borderRadius: 10, overflow: "hidden", border: "1px solid #e5e7eb", background: "#fff" }}>
              <Plot
                data={cpViz.figure?.data || []}
                layout={{
                  autosize: true,
                  // Height will be defined dynamically by the backend figure generation based on rows
                  height: cpViz.figure?.layout?.height || 500, 
                  ...(cpViz.figure?.layout || {}),
                }}
                config={{
                  responsive: true,
                  displaylogo: false,
                  ...(cpViz.config || {}),
                }}
                style={{ width: "100%" }}
                useResizeHandler
              />
            </div>
          )}
        </div>
      )}


      {/* ===== EXTRA FIGURES CAROUSEL ===== */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {hasExtra && currentExtra?.type === "plotly" && (
          <>
            <div style={{ borderRadius: 12, overflow: "hidden", border: "1px solid #e5e7eb", background: "white" }}>
              <Plot
                data={currentExtra.figure?.data || []}
                layout={{
                  autosize: true,
                  height: 260,
                  margin: { l: 60, r: 30, t: 50, b: 50 },
                  ...(currentExtra.figure?.layout || {}),
                }}
                config={{
                  responsive: true,
                  displaylogo: false,
                  ...(currentExtra.config || {}),
                }}
                style={{ width: "100%", height: 260 }}
                useResizeHandler
              />
            </div>

            {extraFigures.length > 1 && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12 }}>
                <button
                  onClick={prevViz}
                  disabled={vizIndex === 0}
                  aria-label="Previous"
                  style={{
                    width: 36, height: 36, borderRadius: "50%", border: "none",
                    background: "#f3f4f6", cursor: vizIndex === 0 ? "default" : "pointer",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    opacity: vizIndex === 0 ? 0.3 : 1,
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="#111827" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 18l-6-6 6-6" />
                  </svg>
                </button>

                <div style={{ display: "flex", gap: 6 }}>
                  {extraFigures.map((viz, i) => (
                    <button
                      key={viz.key}
                      onClick={() => setVizIndex(i)}
                      aria-label={`Go to ${i + 1}`}
                      style={{
                        width: 8, height: 8, padding: 0, borderRadius: "50%", border: "none",
                        background: i === vizIndex ? "#111827" : "#d1d5db", cursor: "pointer",
                      }}
                    />
                  ))}
                </div>

                <button
                  onClick={nextViz}
                  disabled={vizIndex === extraFigures.length - 1}
                  aria-label="Next"
                  style={{
                    width: 36, height: 36, borderRadius: "50%", border: "none",
                    background: "#f3f4f6", cursor: vizIndex === extraFigures.length - 1 ? "default" : "pointer",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    opacity: vizIndex === extraFigures.length - 1 ? 0.3 : 1,
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="#111827" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 6l6 6-6 6" />
                  </svg>
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}