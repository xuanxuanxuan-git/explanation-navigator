import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import InstanceEditor from "./InstanceEditor";
import {
  generateShapBarPlot,
  generateCounterfactualExplanation,
  generateAllCpPlots,
} from "../api.js";

// Helper to standardise the Plotly object
function normaliseVisualisation(viz, fallbackTitle = "Visualisation") {
  if (!viz) return null;
  if (viz.type === "plotly") {
    return { ...viz, title: viz.title || viz.figure?.layout?.title?.text || fallbackTitle };
  }
  if (viz.visualisation?.type === "plotly") {
    return {
      type: "plotly",
      title: viz.visualisation?.title || viz.visualisation?.figure?.layout?.title?.text || fallbackTitle,
      figure: {
        data: viz.visualisation?.figure?.data || [],
        layout: viz.visualisation?.figure?.layout || {},
      },
      config: viz.visualisation?.config || { responsive: false },
      meta: viz.visualisation?.meta || {},
    };
  }
  return null;
}

export default function ExplanationsDashboard() {
  const queryParameters = new URLSearchParams(window.location.search);
  const instanceIdString = queryParameters.get("id");
  const instanceId = instanceIdString ? parseInt(instanceIdString, 10) : 57;

  const [localViz, setLocalViz] = useState(null);
  const [cfViz, setCfViz] = useState(null);
  const [iceViz, setIceViz] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function fetchExplanations() {
      setLoading(true);
      setError("");
      try {
        const [localRes, cfRes, iceRes] = await Promise.all([
          generateShapBarPlot(instanceId),
          generateCounterfactualExplanation(instanceId, 50),
          generateAllCpPlots(instanceId),
        ]);

        if (cancelled) return;

        setLocalViz(normaliseVisualisation(localRes, "Local Feature Importance"));
        setCfViz(normaliseVisualisation(cfRes, "Counterfactual Explanation"));
        setIceViz(normaliseVisualisation(iceRes, "ICE / Ceteris Paribus Plot"));
      } catch (err) {
        if (!cancelled) setError(err.message || "Failed to load explanations.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchExplanations();
    return () => { cancelled = true; };
  }, [instanceId]);

  return (
    <div
      style={{
        backgroundColor: "#f2f2f2",
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center", // Centers the dashboard on ultra-wide screens
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 14,
          padding: 16,
          width: "100%",
          maxWidth: 1000, // Prevents infinite stretching on large monitors
          height: "952px",
          boxSizing: "border-box",
        }}
      >
        {/* ERROR / LOADING ALERTS */}
        {loading && <div style={{ color: "#64748b" }}>Loading dashboard explanations...</div>}
        {error && <div style={{ color: "#991b1b", background: "#fee2e2", padding: 12, borderRadius: 8 }}>{error}</div>}

        {/* TOP: Applicant Profile - Outer Box */}
        <div style={{ 
          flex: "0 0 auto", 
          width: "100%", 
          backgroundColor: "#fff", 
          borderRadius: 8, 
          paddingTop: 8, 
          border: "1px solid #e2e8f0",
          boxSizing: "border-box",
          display: "flex", 
          justifyContent: "center",
        }}>
          {/* Inner Box to constrain the Editor width to its original size */}
          <div style={{
            width: "100%",
            maxWidth: 550,
          }}>
            <InstanceEditor instanceId={instanceId} />
          </div>
        </div>

        {/* BOTTOM: Split layout for explanations */}
        {!loading && !error && (
          <div style={{ display: "flex", gap: 14, flex: 1, minHeight: 0 }}>

            {/* LEFT COLUMN: Local Feature (Top) & Counterfactual (Bottom) */}

            <div style={{ flex: "1", display: "flex", flexDirection: "column", gap: 16, minWidth: 400 }}>

              {/* Local Feature Importance */}
              <div style={{ flex: 0.8, background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0", padding: 6, display: "flex", flexDirection: "column" }}>
                {localViz && (
                  <div style={{ flex: 1, overflow: "hidden" }}>
                    <Plot
                      data={localViz.figure?.data || []}
                      layout={{
                        autosize: true,
                        // Tighter margins, more space on the left for long labels
                        margin: { l: 150, r: 10, t: 40, b: 30 },
                        ...localViz.figure?.layout,
                      }}
                      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                )}
              </div>

              {/* Counterfactual */}
              <div style={{ flex: 0.8, background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0", padding: 6, display: "flex", flexDirection: "column" }}>
                {cfViz && (
                  <div style={{ flex: 1, overflow: "hidden" }}>
                    <Plot
                      data={cfViz.figure?.data || []}
                      layout={{
                        autosize: true,
                        // Tight margins
                        margin: { l: 40, r: 10, t: 40, b: 30 },
                        ...cfViz.figure?.layout,
                      }}
                      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT COLUMN: ICE / CP Plots */}
            <div style={{ flex: "1", background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0", padding: 6, display: "flex", flexDirection: "column", overflowY: "hidden", minWidth: 400 }}>
              {iceViz && (
                <div style={{ flex: 1 }}>
                  <Plot
                    data={iceViz.figure?.data || []}
                    layout={{
                      autosize: true,
                      // Override the backend height to make it shorter and fit the screen better
                      height: iceViz.figure?.layout?.height,
                      margin: { l: 40, r: 20, t: 0, b: 40 },
                      ...iceViz.figure?.layout
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    style={{ width: "100%", height: "100%" }}
                    useResizeHandler
                  />
                </div>
              )}
            </div>

          </div>
        )}
      </div>
    </div>
  );
}