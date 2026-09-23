import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { summarizeDocument } from "../api";

export default function SummarizeTab({ documents }) {
  const [selectedFilename, setSelectedFilename] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  React.useEffect(() => {
    if (documents.length > 0 && !selectedFilename) {
      setSelectedFilename(documents[0].filename);
    }
  }, [documents]);

  async function handleSummarize() {
    setError(null);
    const doc = documents.find((d) => d.filename === selectedFilename);
    if (!doc) return;

    setLoading(true);
    try {
      const res = await summarizeDocument(doc.doc_id, doc.filename);
      setResult(res);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  if (documents.length === 0) {
    return <p className="empty-state">Upload and process a document first.</p>;
  }

  return (
    <div>
      <h2>Summarize a single document</h2>
      <div className="card">
        <select value={selectedFilename} onChange={(e) => setSelectedFilename(e.target.value)}>
          {documents.map((doc) => (
            <option key={doc.doc_id} value={doc.filename}>
              {doc.filename}
            </option>
          ))}
        </select>
        <div className="btn-row" style={{ marginTop: 12 }}>
          <button className="btn" onClick={handleSummarize} disabled={loading}>
            {loading ? "Summarizing..." : "Summarize Document"}
          </button>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>📄 Summary</h3>
          <div className="answer-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary}</ReactMarkdown>
          </div>
          {result.citations && result.citations.length > 0 && (
            <>
              <h3>📌 Sources</h3>
              <ul className="citation-list">
                {result.citations.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
