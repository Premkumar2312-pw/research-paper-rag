import React, { useState } from "react";

/**
 * Expandable panel showing exactly which chunks were retrieved and passed
 * to the LLM for a given answer -- useful for demonstrating the RAG
 * architecture during a viva and for spot-checking grounding.
 */
export default function DebugPanel({ chunks }) {
  const [open, setOpen] = useState(false);

  if (!chunks || chunks.length === 0) return null;

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div
        onClick={() => setOpen(!open)}
        style={{ cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
      >
        <strong style={{ fontSize: "0.9rem" }}>🔍 View Retrieved Chunks ({chunks.length})</strong>
        <span style={{ color: "var(--text-dim)" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{ marginTop: 10 }}>
          <p className="empty-state">
            These are the exact passages retrieved and passed to the LLM to generate the answer above.
          </p>
          {chunks.map((chunk, i) => {
            const scoreBits = [];
            if (chunk.rerank_score != null) scoreBits.push(`rerank: ${chunk.rerank_score.toFixed(3)}`);
            if (chunk.semantic_score != null) scoreBits.push(`semantic: ${chunk.semantic_score.toFixed(3)}`);
            if (chunk.bm25_score != null) scoreBits.push(`bm25: ${chunk.bm25_score.toFixed(3)}`);
            const scoreLabel = scoreBits.length ? scoreBits.join(" · ") : `score: ${(chunk.score || 0).toFixed(3)}`;

            return (
              <div className="chunk-item" key={i}>
                <div className="meta">
                  <strong>{i + 1}. {chunk.filename} — {chunk.location_label}</strong>
                  <div>{scoreLabel}</div>
                </div>
                <div className="text">
                  {chunk.text.length > 600 ? chunk.text.slice(0, 600) + "..." : chunk.text}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
