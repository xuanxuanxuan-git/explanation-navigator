import React, { useEffect, useMemo, useRef, useState } from "react"
import Plot from "react-plotly.js"
import { generateShapBarPlot } from "../api.js"

/* ---------- HELPERS ---------- */

function normaliseVisualisation(viz, fallbackTitle = "Visualisation") {
  if (!viz) return null

  // Direct plotly object: { type: "plotly", figure: { data, layout }, ... }
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

  // Nested: { visualisation: { type: "plotly", ... } }
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
    }
  }

  return null
}

/* ---------- MAIN COMPONENT ---------- */

export default function FigureViewer({ instanceId, visualisations = [] }) {
  /* SHAP state */
  const [shapViz, setShapViz] = useState(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState("")

  /* Extra figures carousel state */
  const [vizIndex, setVizIndex] = useState(0)
  const prevExtraCountRef = useRef(0)

  /* ---------- LOAD SHAP PLOT FOR INSTANCE ---------- */

  useEffect(() => {
    async function loadShap() {
      setShapLoading(true)
      setShapError("")
      try {
        const res = await generateShapBarPlot(instanceId)
        const norm = normaliseVisualisation(
          res,
          `Local feature attribution for applicant ${instanceId}`,
        )
        setShapViz(norm)
      } catch (e) {
        setShapError(e.message || "Failed to load SHAP visualisation")
        setShapViz(null)
      } finally {
        setShapLoading(false)
      }
    }

    loadShap()
  }, [instanceId])

  /* ---------- EXTRA FIGURES (FROM CHAT) ---------- */

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

  // When new extra figures arrive, jump to latest; clamp index otherwise
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

  return (
    <div
      style={{
        padding: 10,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        // height: "100%",
        boxSizing: "border-box",
        overflowY: "auto",
      }}
    >
      {/* ===== SHAP SECTION (OWN SPACE) ===== */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          borderBottom: "1px solid #e5e7eb",
          paddingBottom: 8,
        }}
      >

        {shapLoading && !shapViz && (
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Loading SHAP explanation…
          </div>
        )}

        {shapError && (
          <div
            style={{
              padding: 8,
              borderRadius: 8,
              background: "#fee2e2",
              color: "#991b1b",
              fontSize: 12,
            }}
          >
            {shapError}
          </div>
        )}

        {shapViz?.type === "plotly" && (
          <div
            style={{
              borderRadius: 10,
              overflow: "hidden",
              border: "1px solid #e5e7eb",
              background: "#fff",
            }}
          >
            <Plot
              data={shapViz.figure?.data || []}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 140, r: 20, t: 55, b: 40 },
                ...(shapViz.figure?.layout || {}),
              }}
              config={{
                responsive: true,
                displaylogo: false,
                ...(shapViz.config || {}),
              }}
              style={{ width: "100%", height: 320 }}
              useResizeHandler
            />
          </div>
        )}
      </div>

      {/* ===== EXTRA FIGURES SECTION (CAROUSEL) ===== */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 10,
          flex: 1,
          minHeight: 0,
        }}
      >

        {hasExtra && currentExtra?.type === "plotly" && (
          <>

            {/* Plot */}
            <div
              style={{
                borderRadius: 12,
                overflow: "hidden",
                border: "1px solid #e5e7eb",
                background: "white",
                flex: 1,
                minHeight: 0,
              }}
            >
              <Plot
                data={currentExtra.figure?.data || []}
                layout={{
                  autosize: true,
                  height: 320,
                  margin: { l: 60, r: 30, t: 50, b: 50 },
                  ...(currentExtra.figure?.layout || {}),
                }}
                config={{
                  responsive: true,
                  displaylogo: false,
                  ...(currentExtra.config || {}),
                }}
                style={{ width: "100%", height: 320 }}
                useResizeHandler
              />
            </div>

            {/* Navigation Controls */}
            {extraFigures.length > 1 && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 12,
                }}
              >
                {/* Left */}
                <button
                  onClick={prevViz}
                  disabled={vizIndex === 0}
                  aria-label="Previous visualisation"
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: "50%",
                    border: "none",
                    background: "#f3f4f6",
                    cursor:
                      vizIndex === 0 ? "default" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: vizIndex === 0 ? 0.3 : 1,
                    transition: "all 0.2s",
                  }}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="#111827"
                    strokeWidth={3}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M15 18l-6-6 6-6"
                    />
                  </svg>
                </button>

                {/* Dots */}
                <div style={{ display: "flex", gap: 6 }}>
                  {extraFigures.map((viz, i) => (
                    <button
                      key={viz.key}
                      onClick={() => setVizIndex(i)}
                      aria-label={`Go to visualisation ${i + 1}`}
                      style={{
                        width: 8,
                        height: 8,
                        padding: 0,
                        borderRadius: "50%",
                        border: "none",
                        background:
                          i === vizIndex
                            ? "#111827"
                            : "#d1d5db",
                        transition: "all 0.2s",
                        cursor: "pointer",
                      }}
                    />
                  ))}
                </div>

                {/* Right */}
                <button
                  onClick={nextViz}
                  disabled={vizIndex === extraFigures.length - 1}
                  aria-label="Next visualisation"
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: "50%",
                    border: "none",
                    background: "#f3f4f6",
                    cursor:
                      vizIndex === extraFigures.length - 1
                        ? "default"
                        : "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity:
                      vizIndex === extraFigures.length - 1
                        ? 0.3
                        : 1,
                    transition: "all 0.2s",
                  }}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="#111827"
                    strokeWidth={3}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9 6l6 6-6 6"
                    />
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