import React, { useEffect, useMemo, useState } from "react"
import Plot from "react-plotly.js"
import { fetchInstance, predictInstanceWithChanges } from "../api.js"

export default function InstanceEditor({ instanceId, visualisations = [] }) {

  const [currentInstanceId, setCurrentInstanceId] = useState(instanceId)
  const [loading, setLoading] = useState(false)
  const [features, setFeatures] = useState({})
  const [originalFeatures, setOriginalFeatures] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [originalPrediction, setOriginalPrediction] = useState(null)
  const [error, setError] = useState("")

  /* ---------- VISUALISATION HISTORY ---------- */

  const [vizHistory, setVizHistory] = useState([])
  const [vizIndex, setVizIndex] = useState(0)

  useEffect(() => {
    if (!visualisations || visualisations.length === 0) return

    setVizHistory(prev => {
      const updated = [...prev, ...visualisations]
      setVizIndex(updated.length - 1)
      return updated
    })
  }, [visualisations])

  function prevViz() {
    setVizIndex(i => Math.max(0, i - 1))
  }

  function nextViz() {
    setVizIndex(i => Math.min(vizHistory.length - 1, i + 1))
  }

  const currentViz = vizHistory[vizIndex]

  /* ---------- INSTANCE LOADING ---------- */

  async function loadInstance(id) {
    setLoading(true)
    setError("")
    try {
      const res = await fetchInstance(id)
      const data = res?.data || {}
      const f = data.instance_features || {}
      const p = data.predicted_price

      setFeatures(f)
      setOriginalFeatures(f)
      setPrediction(p)
      setOriginalPrediction(p)
    } catch (e) {
      setError(e.message || "Failed to load instance")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    setCurrentInstanceId(instanceId)
    loadInstance(instanceId)
  }, [instanceId])

  function handleFeatureChange(name, value) {
    setFeatures(prev => ({
      ...prev,
      [name]: value === "" ? "" : Number(value)
    }))
  }

  const changedFields = useMemo(() => {
    const out = {}
    for (const key of Object.keys(features)) {
      if (
        features[key] !== "" &&
        Number(features[key]) !== Number(originalFeatures[key])
      ) {
        out[key] = Number(features[key])
      }
    }
    return out
  }, [features, originalFeatures])

  async function handleRecalculate() {
    if (!Object.keys(changedFields).length) return

    setLoading(true)
    setError("")
    try {
      const res = await predictInstanceWithChanges({
        instance_id: instanceId,
        changes: changedFields
      })
      const data = res?.data || {}
      setPrediction(data.new_prediction)
    } catch (e) {
      setError(e.message || "Prediction failed")
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFeatures(originalFeatures)
    setPrediction(originalPrediction)
  }

  return (
    <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 1 }}>

      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "#fff",
        padding: 8,
        borderRadius: 8,
        border: "1px solid #e5e7eb"
      }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>Instance</span>

        <input
          type="number"
          value={instanceId}
          readOnly
          style={{
            width: 70,
            padding: "4px 6px",
            borderRadius: 6,
            border: "1px solid #d1d5db",
            background: "#f9fafb"
          }}
        />

        <div style={{ marginLeft: "auto", fontSize: 13 }}>
          <b>{prediction ? prediction.toFixed(4) : "—"}</b>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: 8,
          borderRadius: 8,
          background: "#fee2e2",
          color: "#991b1b",
          fontSize: 12
        }}>
          {error}
        </div>
      )}

      {/* Feature Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 8,
        maxHeight: 260,
        overflowY: "auto",
        padding: 6,
        background: "#fff",
        borderRadius: 8,
        border: "1px solid #e5e7eb"
      }}>
        {Object.entries(features).map(([name, value]) => (
          <div key={name} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <label style={{ fontSize: 11, color: "#6b7280" }}>{name}</label>

            <input
              type="number"
              step="any"
              value={value}
              onChange={(e) => handleFeatureChange(name, e.target.value)}
              style={{
                padding: "4px 6px",
                borderRadius: 6,
                border: "1px solid #d1d5db",
                fontSize: 12,
                background:
                  Number(value) !== Number(originalFeatures[name])
                    ? "#eef4ff"
                    : "#fff"
              }}
            />
          </div>
        ))}
      </div>

      {/* Buttons */}
      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={handleRecalculate}
          disabled={loading || !Object.keys(changedFields).length}
          style={{
            flex: 1,
            padding: "8px",
            borderRadius: 8,
            border: "none",
            background: "#2563eb",
            color: "white",
            fontSize: 13,
            cursor: "pointer",
            opacity: loading ? 0.6 : 1
          }}
        >
          Recalculate
        </button>

        <button
          onClick={handleReset}
          disabled={loading}
          style={{
            padding: "8px",
            borderRadius: 8,
            border: "1px solid #d1d5db",
            background: "#fff",
            fontSize: 13,
            cursor: "pointer"
          }}
        >
          Reset
        </button>
      </div>
    
    {/* ---------- VISUALISATION VIEWER ---------- */}

    {vizHistory.length > 0 && currentViz?.type === "plotly" && (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 20 }}>

        {/* Plot */}
        <div
        style={{
            borderRadius: 12,
            overflow: "hidden",
            border: "1px solid #e5e7eb",
            background: "white"
        }}
        >
        <Plot
            data={currentViz.figure?.data || []}
            layout={{ ...(currentViz.figure?.layout || {}), autosize: true }}
            config={currentViz.config || { responsive: true }}
            style={{ width: "100%", height: 360 }}
            useResizeHandler
        />
        </div>

        {/* Navigation Controls */}
        <div
        style={{
            display: "flex",
            alignItems: "center", // vertically center buttons with dots
            justifyContent: "center",
            gap: 12,
        }}
        >
        {/* Left */}
        <button
            onClick={prevViz}
            disabled={vizIndex === 0}
            style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            border: "none",
            background: "#f3f4f6",
            cursor: vizIndex === 0 ? "default" : "pointer",
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
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 18l-6-6 6-6" />
            </svg>
        </button>

        {/* Dots */}
        <div style={{ display: "flex", gap: 6 }}>
            {vizHistory.map((_, i) => (
            <div
                key={i}
                style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: i === vizIndex ? "#111827" : "#d1d5db",
                transition: "all 0.2s",
                }}
            />
            ))}
        </div>

        {/* Right */}
        <button
            onClick={nextViz}
            disabled={vizIndex === vizHistory.length - 1}
            style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            border: "none",
            background: "#f3f4f6",
            cursor: vizIndex === vizHistory.length - 1 ? "default" : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: vizIndex === vizHistory.length - 1 ? 0.3 : 1,
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
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 6l6 6-6 6" />
            </svg>
        </button>
        </div>

    </div>
    )}
    </div>
  )
}