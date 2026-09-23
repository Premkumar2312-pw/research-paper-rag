import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { askQuestion } from "../api";
import ConfidenceBadge from "./ConfidenceBadge";
import DebugPanel from "./DebugPanel";

const EXAMPLE_QUESTIONS = [
  "What is the main contribution of this paper?",
  "What dataset was used?",
  "What methodology was proposed?",
  "What are the limitations?",
  "Compare the methodologies used in these documents.",
];

export default function AskTab({ documents, selectedIds, onAnswered }) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function handleAsk() {
    setError(null);
    if (documents.length === 0) {
      setError("No documents have been uploaded and processed yet.");
      return;
    }
    if (selectedIds.length === 0) {
      setError("Please select at least one active document to search.");
      return;
    }
    if (!question.trim()) {
      setError("Please type a question first.");
      return;
    }

    setLoading(true);
    try {
      const res = await askQuestion(question, selectedIds, topK);
      setResult(res);
      onAnswered({
        question,
        answer: res.answer,
        citations: res.citations,
        confidence: res.confidence,
      });
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Ask a question about your selected documents</h2>

      <div style={{ marginBottom: 12 }}>
        {EXAMPLE_QUESTIONS.map((q, i) => (
          <span className="example-chip" key={i} onClick={() => setQuestion(q)}>
            {q}
          </span>
        ))}
      </div>

      <div className="card">
        <textarea
          rows={3}
          placeholder="Type your question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

        <div className="slider-row" style={{ marginTop: 10 }}>
          <span>Retrieval depth (chunks passed to LLM): <strong>{topK}</strong></span>
          <input
            type="range"
            min="2"
            max="10"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </div>

        <div className="btn-row" style={{ marginTop: 12 }}>
          <button className="btn" onClick={handleAsk} disabled={loading}>
            {loading ? "Thinking..." : "Ask Question"}
          </button>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>🧠 Answer</h3>
          {result.is_comparison && (
            <p className="empty-state">📊 Comparison view — structured across the selected documents.</p>
          )}
          <div className="answer-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
          </div>

          <div style={{ margin: "10px 0" }}>
            <ConfidenceBadge confidence={result.confidence} />
            <p className="empty-state" style={{ marginTop: 6 }}>
              Estimated from how closely retrieved evidence matches your question —
              not a guarantee the answer is factually correct. Please verify against sources.
            </p>
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

          <DebugPanel chunks={result.chunks_used} />
        </div>
      )}
    </div>
  );
}
