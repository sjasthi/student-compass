# Student Compass — End User Guide

Welcome to **Student Compass**, your AI-powered assistant for navigating Metropolitan State University information.  
This guide is written for **end users** — students, staff, or anyone using the web application — and explains how to use the system without needing any technical background.

---

## What Student Compass Does

Student Compass helps you:

- Ask questions about university topics (admissions, financial aid, registration, policies, etc.)
- Receive accurate, context‑aware answers based on official documents
- View the sources used to generate each answer
- Upload documents 
- Run evaluations 

The system uses:

- **Google Cloud Storage (GCS)** to store documents  
- **ChromaDB** to index and search document content  
- **Gemini 2.5 Flash** to generate helpful answers  

---

## Using the Chat Interface (Home Page)

The Home page is the main place where you interact with Student Compass.

### How to Ask a Question

1. Type your question into the text box at the bottom.
2. Press **Enter** or click **Ask**.
3. The AI will begin responding immediately.
4. Answers appear one token at a time (streaming), just like a live chat.

### Viewing Sources

Below each answer, you’ll see a **Sources** section showing:

- The document name  
- The document type (e.g., Admissions, Financial Aid)  
- A short summary (if available)  

This helps you verify where the information came from.

### Starting a New Conversation

Click **New Conversation** at the top of the page to clear your chat history.

---

## Admin Page — Document Management

The Admin page lets you manage the knowledge base.

### Uploading a File

You can upload:

- PDF  
- DOCX / DOC  
- TXT  
- MD (Markdown)  
- Web pages (via URL)

When you upload a file:

1. It is stored in Google Cloud Storage.
2. The system automatically processes and indexes it.
3. It becomes searchable in the chat within a few seconds.

### Replacing an Existing File

Enable **Replace Existing** to overwrite older versions of the same document.

### Deleting a File

Click **Delete** next to any file to remove it from:

- GCS  
- The search index  

### Manual Sync

If something seems out of sync, click **Sync** to force the system to:

- Add missing files  
- Remove deleted files  
- Rebuild the index if needed  

---

## Test Page — Evaluation Runner

The Test page allow users to run accuracy evaluations.

You can:

- Choose evaluation modes (RAG, keyword-only, prompt-only)
- Adjust parameters like chunk size, top‑k, temperature, and top‑p
- Watch live progress through streaming updates
- Download results as a CSV file

This is mainly for quality assurance and tuning.

---

## Troubleshooting

### The system says the knowledge base is empty

This usually means the backend index needs to be rebuilt.  
Admins can fix this by clicking **Sync** on the Admin page.

### A document isn’t showing up in search

Make sure:

- The upload finished successfully  
- The file type is supported  
- The document is marked as **active** in the Admin list  

### Answers seem incomplete

Try rephrasing your question or asking something more specific.

---

## Support

If you encounter issues or have suggestions, contact your system administrator or project maintainer.

Student Compass is designed to make university information easier to access — we hope it helps you find what you need quickly and confidently.
