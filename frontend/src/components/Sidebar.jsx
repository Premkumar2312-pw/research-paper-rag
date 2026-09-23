import React, { useRef, useState } from "react";
import { uploadDocuments, clearAllDocuments } from "../api";

const FILE_TYPE_ICONS = {
  pdf: "📕", docx: "📄", pptx: "📊", txt: "📝", md: "📝", csv: "📈", xlsx: "📈",
};

const SUPPORTED_EXTENSIONS = ["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx"];

export default function Sidebar({ documents, selectedIds, onToggleSelect, onDocumentsChanged }) {
  const fileInputRef = useRef(null);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [messages, setMessages] = useState([]); // {type: 'success'|'error', text}

  function handleFileChange(e) {
    setPendingFiles(Array.from(e.target.files));
  }

  async function handleProcess() {
    if (pendingFiles.length === 0) return;
    setProcessing(true);
    setMessages([]);
    try {
      const result = await uploadDocuments(pendingFiles);
      const newMessages = [];
      for (const doc of result.processed) {
        const icon = FILE_TYPE_ICONS[doc.file_type] || "📄";
        let text = `${icon} ${doc.filename} (${doc.file_type.toUpperCase()}): ${doc.segment_count} segment(s), ${doc.chunk_count} chunks indexed`;
        if (doc.ocr_segments_used) text += ` (${doc.ocr_segments_used} page(s) used OCR)`;
        newMessages.push({ type: "success", text });
      }
      for (const err of result.errors) {
        newMessages.push({ type: "error", text: `${err.filename}: ${err.error}` });
      }
      setMessages(newMessages);
      setPendingFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
      onDocumentsChanged();
    } catch (err) {
      setMessages([{ type: "error", text: err.message }]);
    } finally {
      setProcessing(false);
    }
  }

  async function handleClearAll() {
    if (!window.confirm("Remove all uploaded documents? This can't be undone.")) return;
    await clearAllDocuments();
    onDocumentsChanged();
  }

  return (
    <div className="sidebar">
      <h1>📚 Research Paper Intelligence</h1>
      <p className="subtitle">Lightweight RAG for research papers &amp; documents</p>

      <div className="card">
        <h2>📤 Upload Documents</h2>
        <p className="empty-state" style={{ marginBottom: 8 }}>
          Supported: PDF, DOCX, PPTX, TXT, MD, CSV, XLSX
        </p>
        <label className="file-drop">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={SUPPORTED_EXTENSIONS.map((e) => "." + e).join(",")}
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          {pendingFiles.length > 0
            ? `${pendingFiles.length} file(s) selected`
            : "Click to choose files"}
        </label>

        <div style={{ marginTop: 10 }}>
          <button
            className="btn"
            style={{ width: "100%" }}
            disabled={pendingFiles.length === 0 || processing}
            onClick={handleProcess}
          >
            {processing ? "Processing..." : "Process Documents"}
          </button>
        </div>

        {messages.map((m, i) => (
          <div key={i} className={m.type === "success" ? "success-box" : "error-box"} style={{ marginTop: 8 }}>
            {m.text}
          </div>
        ))}
      </div>

      <div className="card">
        <h2>📑 Active Documents</h2>
        {documents.length === 0 ? (
          <p className="empty-state">No documents indexed yet.</p>
        ) : (
          <>
            <p className="empty-state" style={{ marginBottom: 6 }}>Select which to search:</p>
            {documents.map((doc) => (
              <label className="doc-item" key={doc.doc_id}>
                <input
                  type="checkbox"
                  checked={selectedIds.includes(doc.doc_id)}
                  onChange={() => onToggleSelect(doc.doc_id)}
                />
                <span className="name">
                  {FILE_TYPE_ICONS[doc.file_type] || "📄"} {doc.filename}
                </span>
                <span className="doc-type-badge">{doc.file_type}</span>
              </label>
            ))}
            <div style={{ marginTop: 12 }}>
              <button className="btn danger" style={{ width: "100%" }} onClick={handleClearAll}>
                🗑️ Clear All Documents
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
