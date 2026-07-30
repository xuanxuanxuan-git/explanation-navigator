import React, { useEffect } from 'react'
import ChatPage from './components/ChatPage.jsx'
import ExplanationsDashboard from './components/Interactive.jsx'

export default function App() {
  // removes the default white border around the entire webpage
  useEffect(() => {
    document.body.style.margin = "0";
    document.body.style.padding = "0";
  }, []);
  // Check the current URL path
  const path = window.location.pathname;

  // If the user navigates to /dashboard, render ONLY the dashboard
  if (path === '/dashboard') {
    return (
      <div style={{ fontFamily: 'system-ui, Arial', margin: 0, padding: 0 }}>
        <ExplanationsDashboard />
      </div>
    );
  }

  // Otherwise, render your normal Chat app
  return (
    <div style={{ fontFamily: 'system-ui, Arial', padding: "5px 20px 5px 20px", height: "97vh", boxSizing: "border-box", maxWidth: 1300, minWidth: 900, minHeight: 640, margin: "0 auto"}}>
      {/* <h2 style={{ marginTop:10, marginBottom: 8, marginLeft: 14}}></h2> */}
      <ChatPage />
    </div>
  )
}