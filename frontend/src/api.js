/**
 * api.js
 *
 * Thin fetch wrapper around the FastAPI backend. Every function returns a
 * parsed JSON response and throws a plain Error with the backend's message
 * on failure, so components can just try/catch and show err.message.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handleResponse(response) {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON; keep the generic message
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return handleResponse(res);
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`);
  return handleResponse(res);
}

export async function uploadDocuments(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/api/documents/${docId}`, { method: "DELETE" });
  return handleResponse(res);
}

export async function clearAllDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`, { method: "DELETE" });
  return handleResponse(res);
}

export async function askQuestion(question, docIds, topK) {
  const res = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, doc_ids: docIds, top_k: topK }),
  });
  return handleResponse(res);
}

export async function summarizeDocument(docId, filename) {
  const res = await fetch(`${API_BASE}/api/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId, filename }),
  });
  return handleResponse(res);
}
