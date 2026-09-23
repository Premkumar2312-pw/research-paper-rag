import React from "react";
import ConfidenceBadge from "./ConfidenceBadge";

/**
 * Question/answer history for the current session, plus a one-click
 * export to a downloadable Markdown file -- handy for pasting into a
 * final-year report or keeping a record of a demo session.
 */
export default function HistoryTab({ history }) {
  function handleExport() {
    if (history.length === 0) return;

    const lines = ["# Research Paper Intelligence — Session Q&A Export", ""];
    history
      .slice()
      .reverse()
      .forEach((entry, i) => {
        lines.push(`## ${i + 1}. ${entry.question}`);
        lines.push("");
        lines.push(entry.answer);
        lines.push("");
        lines.push(`_Estimated confidence: ${entry.confidence}_`);
        if (entry.citations && entry.citations.length > 0) {
          lines.push("");
          lines.push("**Sources:**");
          entry.citations.forEach((c) => lines.push(`- ${c}`));
        }
        lines.push("");
        lines.push("---");
        lines.push("");
      });

    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "research-paper-qa-session.md";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Question history (this session)</h2>
        {history.length > 0 && (
          <button className="btn secondary" onClick={handleExport}>
            ⬇️ Export as Markdown
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <p className="empty-state" style={{ marginTop: 12 }}>
          No questions asked yet in this session.
        </p>
      ) : (
        <div className="card" style={{ marginTop: 12 }}>
          {history.map((entry, i) => (
            <details className="history-item" key={i} open={i === 0}>
              <summary>{entry.question}</summary>
              <div style={{ padding: "0 0 14px 4px" }}>
                <p className="answer-body">{entry.answer}</p>
                <ConfidenceBadge confidence={entry.confidence} />
                {entry.citations && entry.citations.length > 0 && (
                  <ul className="citation-list">
                    {entry.citations.map((c, j) => (
                      <li key={j}>{c}</li>
                    ))}
                  </ul>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
