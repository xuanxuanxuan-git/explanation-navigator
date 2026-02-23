import React from 'react'
import Plot from 'react-plotly.js'

export default function VisualisationPanel({ visualisations }) {
  // const [active, setActive] = useState(null) // { plotIndex, traceIndex, pointNumber } etc

  if (!visualisations?.length) {
    return (
      <div style={{ padding: 16, color: '#666' }}>
        Figures will appear here.
      </div>
    )
  }

  return (
    <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      {visualisations.map((v, plotIndex) => {
        if (v?.type !== 'plotly') return null
        const figure = v.figure || {}
        const data = figure.data || []
        const layout = figure.layout || {}
        const config = v.config || { responsive: true }

        return (
          <div key={plotIndex} style={{ border: '1px solid #e5e7eb', borderRadius: 8, background: 'white' }}>
            <div style={{ padding: 10, borderBottom: '1px solid #f1f5f9', fontSize: 12, fontWeight: 600 }}>
              {v?.meta?.tool || `Plot ${plotIndex + 1}`}
            </div>

            <div style={{ padding: 10 }}>
              <Plot
                data={data}
                layout={{ ...layout, autosize: true }}
                config={config}
                style={{ width: '100%', height: 420 }}
                useResizeHandler
                // onClick={(evt) => {
                //   // Plotly click gives points with pointNumber/curveNumber. 
                //   const p = evt?.points?.[0]
                //   if (!p) return
                //   setActive({ plotIndex, traceIndex: p.curveNumber, pointNumber: p.pointNumber })
                // }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}


