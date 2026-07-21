import React, { useEffect, useRef } from 'react'
import TypingStatus from './TypingStatus.jsx'
import ReactMarkdown from 'react-markdown'
import Plot from 'react-plotly.js'

// Reuse Dashboard normalisation logic here to ensure the plot renders correctly
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
    }
  }

  return null
}

export default function MessageList({ messages, busy, status }) {
  const lastUserMessageRef = useRef(null);

  const lastUserIndex = messages.map(m => m.role).lastIndexOf('user');

  useEffect(() => {
    if (lastUserMessageRef.current) {
      lastUserMessageRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'start',   
      });
    }
  }, [lastUserIndex]); 

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: 10, 
    }}>
      {messages.map((m, idx) => {
        const isLastUser = idx === lastUserIndex;

        return (
          <div
            key={idx}
            ref={isLastUser ? lastUserMessageRef : null}
            style={{
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%',
              padding: '10px 12px',
              borderRadius: 12,
              background: m.role === 'user' ? '#e8f0fe' : '#f5f5f5',
              scrollMarginTop: "10px" 
            }}
          >
            <div style={{ fontSize: 12, opacity: 0.65, marginBottom: 4 }}>
              {m.role}
            </div>
            
            <div className="markdown-body" style={{ fontSize: 14 }}>
              {m.role === 'assistant' && !m.content && busy ? (
                <TypingStatus status={status}/>
              ) : (
                <ReactMarkdown>{m.content}</ReactMarkdown>
              )}
            </div>

            {m.visualisations && m.visualisations.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
                {m.visualisations.map((rawViz, vIdx) => {
                  const norm = normaliseVisualisation(rawViz);
                  if (!norm || norm.type !== 'plotly') return null;

                  return (
                    <div 
                      key={vIdx} 
                      style={{
                        width: "85%",           
                        maxWidth: 450,          
                        margin: "0 auto",       
                        borderRadius: 10,
                        overflow: "hidden",
                        border: "1px solid #e5e7eb",
                        background: "#fff",
                      }}
                    >
                      <Plot
                        data={norm.figure?.data || []}
                        layout={{
                          autosize: true,
                          margin: { l: 40, r: 20, t: 40, b: 40 },
                          height: 280,
                          ...(norm.figure?.layout || {}),
                        }}
                        config={{
                          responsive: true,
                          displaylogo: false,
                          ...(norm.config || {}),
                        }}
                        style={{ width: "100%", height: 280 }}
                        useResizeHandler
                      />
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })}
      
      {/* Styles for better Markdown spacing inside chat bubbles */}
      <style>{`
        .markdown-body p:first-of-type { margin-top: 0; }
        .markdown-body p:last-of-type { margin-bottom: 0; }
        .markdown-body ul, .markdown-body ol { margin-top: 4px; margin-bottom: 4px; padding-left: 20px; }
      `}</style>
    </div>
  )
}