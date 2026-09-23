import React from "react";

const ICONS = {
  "High confidence": { icon: "🟢", cls: "high" },
  "Medium confidence": { icon: "🟡", cls: "medium" },
  "Low confidence — verify manually": { icon: "🔴", cls: "low" },
};

/**
 * Displays the estimated retrieval confidence returned by the backend.
 * This reflects how closely the retrieved evidence matched the question --
 * it is NOT a guarantee the generated answer is factually correct.
 */
export default function ConfidenceBadge({ confidence }) {
  const info = ICONS[confidence] || { icon: "🟡", cls: "medium" };
  return (
    <span className={`confidence-badge ${info.cls}`} title="Estimated from retrieval evidence, not a correctness guarantee.">
      {info.icon} {confidence}
    </span>
  );
}
