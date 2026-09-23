import React, { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import AskTab from "./components/AskTab";
import SummarizeTab from "./components/SummarizeTab";
import HistoryTab from "./components/HistoryTab";
import { getHealth, listDocuments } from "./api";

const TABS = [
  { id: "ask", label: "❓ Ask a Question" },
  { id: "summarize", label: "🧾 Summarize a Document" },
  { id: "history", label: "🕘 Question History" },
];

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [activeTab, setActiveTab] = useState("ask");
  const [history, setHistory] = useState([]);
  const [groqConfigured, setGroqConfigured] = useState(true);
  const [backendReachable, setBackendReachable] = useState(true);

  async function refreshDocuments() {
    try {
      const res = await listDocuments();
      setDocuments(res.documents);
      // Keep selection in sync: default new docs to selected, drop removed ones.
      setSelectedIds((prev) => {
        const stillValid = prev.filter((id) => res.documents.some((d) => d.doc_id === id));
        const newIds = res.documents
          .map((d) => d.doc_id)
          .filter((id) => !prev.includes(id));
        return [...stillValid, ...newIds];
      });
    } catch (err) {
      setBackendReachable(false);
    }
  }

  useEffect(() => {
    getHealth()
      .then((h) => {
        setGroqConfigured(h.groq_configured);
        setBackendReachable(true);
      })
      .catch(() => setBackendReachable(false));
    refreshDocuments();
  }, []);

  function handleToggleSelect(docId) {
    setSelectedIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  }

  function handleAnswered(entry) {
    setHistory((prev) => [entry, ...prev]);
  }

  if (!backendReachable) {
    return (
      <div style={{ padding: 40 }}>
        <div className="error-box">
          Could not reach the backend API. Make sure it's running:
          <br />
          <code>cd backend && uvicorn main:app --reload --port 8000</code>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        documents={documents}
        selectedIds={selectedIds}
        onToggleSelect={handleToggleSelect}
        onDocumentsChanged={refreshDocuments}
      />

      <div className="main">
        {!groqConfigured && (
          <div className="warning-box">
            ⚠️ GROQ_API_KEY is not configured on the backend. Add it to{" "}
            <code>backend/.env</code> before asking questions. You can still upload and
            process documents without it.
          </div>
        )}

        <div className="tabs">
          {TABS.map((tab) => (
            <div
              key={tab.id}
              className={`tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </div>
          ))}
        </div>

        {activeTab === "ask" && (
          <AskTab documents={documents} selectedIds={selectedIds} onAnswered={handleAnswered} />
        )}
        {activeTab === "summarize" && <SummarizeTab documents={documents} />}
        {activeTab === "history" && <HistoryTab history={history} />}

        <div style={{ marginTop: 40, color: "var(--text-dim)", fontSize: "0.8rem" }}>
          Research Paper Intelligence — a lightweight RAG system. Answers are grounded
          strictly in the uploaded documents; nothing is invented.
        </div>
      </div>
    </div>
  );
}
