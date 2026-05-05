import React, { useEffect, useMemo, useState } from "react"
import { fetchInstance, predictInstanceWithChanges } from "../api.js"

export default function InstanceEditor({ instanceId }) {
  const [currentInstanceId, setCurrentInstanceId] = useState(instanceId)
  const [loading, setLoading] = useState(false)
  const [features, setFeatures] = useState({})
  const [originalFeatures, setOriginalFeatures] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [originalPrediction, setOriginalPrediction] = useState(null)
  const [error, setError] = useState("")

  // ----------------------------
  // Load instance
  // ----------------------------
  async function loadInstance(id) {
    setLoading(true)
    setError("")
    try {
      const res = await fetchInstance(id)
      const data = res?.data || {}

      const f = data.features || {}
      const p = data.prediction

      setFeatures(f)
      setOriginalFeatures(JSON.parse(JSON.stringify(f))) // deep copy

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

  // ----------------------------
  // Handle slider change
  // ----------------------------
  function handleSliderChange(name, normVal) {
    setFeatures(prev => {
      const f = prev[name]
      const realVal = f.min + normVal * (f.max - f.min)

      return {
        ...prev,
        [name]: {
          ...f,
          value: realVal,
          normalised: normVal,
        },
      }
    })
  }

  // ----------------------------
  // Detect changes
  // ----------------------------
  const changedFields = useMemo(() => {
    const out = {}

    for (const key of Object.keys(features)) {
      const newVal = features[key].value
      const oldVal = originalFeatures[key]?.value

      if (Number(newVal) !== Number(oldVal)) {
        out[key] = Number(newVal)
      }
    }

    return out
  }, [features, originalFeatures])

  // ----------------------------
  // Recalculate prediction
  // ----------------------------
  async function handleRecalculate() {
    if (!Object.keys(changedFields).length) return

    setLoading(true)
    setError("")

    try {
      const res = await predictInstanceWithChanges({
        instance_id: currentInstanceId,
        changes: changedFields,
      })

      const data = res?.data || {}
      setPrediction(data.new_prediction)
    } catch (e) {
      setError(e.message || "Prediction failed")
    } finally {
      setLoading(false)
    }
  }

  // ----------------------------
  // Reset
  // ----------------------------
  function handleReset() {
    setFeatures(JSON.parse(JSON.stringify(originalFeatures)))
    setPrediction(originalPrediction)
  }

  // ----------------------------
  // UI
  // ----------------------------
  return (
    <div
      style={{
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      {/* Error */}
      {error && (
        <div
          style={{
            padding: 8,
            borderRadius: 8,
            background: "#fee2e2",
            color: "#991b1b",
            fontSize: 12,
          }}
        >
          {error}
        </div>
      )}

      {/* Combined card */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          background: "#fff",
          borderRadius: 8,
          border: "1px solid #e5e7eb",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: 8,
            borderBottom: "1px solid #e5e7eb",
            background: "#fff",
            position: "sticky",
            top: 0,
            zIndex: 1,
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>
            Applicant ID: {currentInstanceId}
          </span>

          <div style={{ marginLeft: "auto", fontSize: 13, color: "#111827" }}>
            <span style={{ fontWeight: 600 }}>Default Risk: </span>
            <b>
              {prediction != null ? Number(prediction).toFixed(4) : "—"}
            </b>
          </div>
        </div>

        {/* Feature Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 8,
            maxHeight: 320,
            overflowY: "auto",
            padding: 6,
            background: "#fff",
          }}
        >
          {Object.entries(features).map(([name, f]) => {
            const { value, min, max, normalised } = f

            const originalVal = originalFeatures[name]?.value
            const isChanged = Number(value) !== Number(originalVal)

            return (
              <div
                key={name}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                  padding: 6,
                  borderRadius: 6,
                  background: isChanged ? "#eef4ff" : "#fff",
                }}
              >
                {/* Feature name */}
                <label style={{ fontSize: 11, color: "#111827" }}>
                  {name}
                </label>

                {/* Slider row */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "10px 1fr 20px",
                    alignItems: "center",
                    gap: 6,
                    margin: "5px 0",
                  }}
                >
                  {/* Min */}
                  <span
                    style={{
                      fontSize: 10,
                      color: "#9ca3af",
                      textAlign: "right",
                    }}
                  >
                    {min.toFixed(0)}
                  </span>

                  {/* Slider */}
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.001}
                    value={normalised}
                    onChange={e =>
                      handleSliderChange(name, Number(e.target.value))
                    }
                    style={{
                      WebkitAppearance: "none",
                      appearance: "none",
                      width: "100%",
                      height: "4.1pt",
                      background: "#e5e7eb", /* gray bar */
                      borderRadius: "999px",
                      outline: "none",
                    }}
                  />

                  {/* Max */}
                  <span
                    style={{
                      fontSize: 10,
                      color: "#9ca3af",
                    }}
                  >
                    {max.toFixed(0)}
                  </span>
                </div>

                {/* Current value */}
                <div
                  style={{
                    fontSize: 11,
                    textAlign: "center",
                    color: "#111827",
                  }}
                >
                  {value.toFixed(2)}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Buttons */}
      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={handleRecalculate}
          disabled={
            loading || !Object.keys(changedFields).length
          }
          style={{
            flex: 1,
            padding: "8px",
            borderRadius: 8,
            border: "none",
            background: "#2563eb",
            color: "white",
            fontSize: 13,
            cursor:
              loading ||
                !Object.keys(changedFields).length
                ? "default"
                : "pointer",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? "Loading..." : "Recalculate"}
        </button>

        <button
          onClick={handleReset}
          disabled={loading}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #d1d5db",
            background: "#fff",
            fontSize: 13,
            cursor: loading ? "default" : "pointer",
          }}
        >
          Reset
        </button>
      </div>
    </div >
  )
}