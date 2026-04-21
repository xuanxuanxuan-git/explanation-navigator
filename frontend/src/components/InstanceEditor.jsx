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

  async function loadInstance(id) {
    setLoading(true)
    setError("")
    try {
      const res = await fetchInstance(id)
      const data = res?.data || {}
      const f = data.instance_features || {}
      const p = data.prediction

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
      [name]: value === "" ? "" : Number(value),
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

  function handleReset() {
    setFeatures(originalFeatures)
    setPrediction(originalPrediction)
  }

  return (
    <div
      style={{
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "#fff",
          padding: 8,
          borderRadius: 8,
          border: "1px solid #e5e7eb",
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 600 }}>Applicant ID</span>

        <input
          type="number"
          value={currentInstanceId}
          readOnly
          style={{
            width: 70,
            padding: "4px 6px",
            borderRadius: 6,
            border: "1px solid #d1d5db",
            background: "#f9fafb",
          }}
        />

        <div style={{ marginLeft: "auto", fontSize: 13 }}>
          <b>{prediction != null ? Number(prediction).toFixed(4) : "—"}</b>
        </div>
      </div>

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

      {/* Feature Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
          maxHeight: 260,
          overflowY: "auto",
          padding: 6,
          background: "#fff",
          borderRadius: 8,
          border: "1px solid #e5e7eb",
        }}
      >
        {Object.entries(features).map(([name, value]) => (
          <div
            key={name}
            style={{ display: "flex", flexDirection: "column", gap: 2 }}
          >
            <label style={{ fontSize: 11, color: "#6b7280" }}>{name}</label>

            <input
              type="number"
              step="any"
              value={value}
              onChange={e => handleFeatureChange(name, e.target.value)}
              style={{
                padding: "4px 6px",
                borderRadius: 6,
                border: "1px solid #d1d5db",
                fontSize: 12,
                background:
                  Number(value) !== Number(originalFeatures[name])
                    ? "#eef4ff"
                    : "#fff",
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
            cursor:
              loading || !Object.keys(changedFields).length
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
    </div>
  )
}