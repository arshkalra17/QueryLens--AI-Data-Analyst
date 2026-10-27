/* QueryLens — frontend SPA (no build step). */
(() => {
  "use strict";

  const API = "/api";
  const SAMPLE_FILES = ["sales_data.csv", "retail_transactions.csv", "web_traffic.csv"];

  const state = {
    sessionId: null,
    datasets: [],
    documents: [],
    pinned: null,
    streaming: false,
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // ---------------------------------------------------------------- helpers
  function toast(msg, isErr = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.toggle("err", isErr);
    t.classList.add("show");
    clearTimeout(t._timer);
    t._timer = setTimeout(() => t.classList.remove("show"), 3200);
  }

  async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }
    const res = await fetch(API + path, { ...options, headers });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || (data.errors && data.errors.length ? data.errors[0].error : res.statusText);
      } catch (_) { /* ignore */ }
      throw new Error(`${res.status}: ${detail}`);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("json")) return res.json();
    return res;
  }

  // ---------------------------------------------------------------- sessions
  async function ensureSession() {
    if (state.sessionId) return;
    const data = await api("/chat/session", { method: "POST", body: JSON.stringify({ dataset_ids: state.datasets.map((d) => d.dataset_id) }) });
    state.sessionId = data.session_id;
  }

  function resetSession() {
    state.sessionId = null;
  }

  // ---------------------------------------------------------------- health
  async function refreshHealth() {
    try {
      const h = await api("/health");
      const dot = $("#status-dot");
      dot.className = "dot ok";
      $("#status-text").textContent = "online";
      const llm = h.llm;
      $("#provider-badge").textContent = llm.configured ? `LLM: ${llm.provider}/${llm.model}` : "rule-based engine (no LLM key)";
      if (h.demo && h.demo.enabled) {
        $("#demo-banner").hidden = false;
        const n = h.demo.rate_limit_per_hour;
        $("#demo-limit").textContent = n ? `Limit: ${n} requests per hour.` : "";
      }
    } catch (e) {
      $("#status-dot").className = "dot err";
      $("#status-text").textContent = "offline";
    }
  }

  // ---------------------------------------------------------------- datasets
  async function refreshDatasets() {
    const data = await api("/datasets");
    state.datasets = data.datasets;
    if (state.pinned && !state.datasets.find((d) => d.dataset_id === state.pinned)) state.pinned = null;
    if (!state.pinned && state.datasets.length) state.pinned = state.datasets[0].dataset_id;
    renderDatasetList();
  }

  function renderDatasetList() {
    const list = $("#dataset-list");
    list.innerHTML = "";
    $("#dataset-count").textContent = state.datasets.length;
    if (!state.datasets.length) {
      list.appendChild(el("div", "dataset-empty", "No datasets yet — drop a CSV above."));
      return;
    }
    for (const ds of state.datasets) {
      const item = el("div", "dataset-item" + (ds.dataset_id === state.pinned ? " active" : ""));
      const row = el("div", "ds-name");
      row.appendChild(el("span", "", ds.name));
      const del = el("button", "ds-del", "✕");
      del.title = "Remove dataset";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          await api(`/datasets/${ds.dataset_id}`, { method: "DELETE" });
          toast(`Removed ${ds.name}`);
          await refreshDatasets();
        } catch (err) { toast(err.message, true); }
      });
      row.appendChild(del);
      item.appendChild(row);
      item.appendChild(el("div", "ds-meta", `${ds.row_count.toLocaleString()} rows · ${ds.col_count} cols`));
      const cols = ds.columns.slice(0, 8).map((c) => c.name).join(", ") + (ds.columns.length > 8 ? "…" : "");
      item.appendChild(el("div", "ds-cols", cols));
      item.addEventListener("click", async () => {
        state.pinned = ds.dataset_id;
        renderDatasetList();
        if (state.sessionId) {
          try { await api(`/chat/session/${state.sessionId}/reset`, { method: "POST" }); } catch (_) { /* ignore */ }
          toast(`Pinned ${ds.name}. Fresh context started.`);
        }
      });
      list.appendChild(item);
    }
  }

  async function uploadFiles(fileList) {
    if (!fileList || !fileList.length) return;
    const fd = new FormData();
    for (const f of fileList) fd.append("files", f);
    toast(`Uploading ${fileList.length} file(s)…`);
    try {
      const res = await fetch(API + "/datasets", { method: "POST", body: fd });
      const data = await res.json();
      const ok = data.imported || [];
      const bad = data.errors || [];
      if (ok.length) {
        await ensureSession();
        toast(`Imported ${ok.length} dataset(s).`);
      }
      if (bad.length) toast(`Failed: ${bad[0].error || bad[0].file}`, true);
      await refreshDatasets();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function loadSamples() {
    try {
      toast("Loading sample datasets…");
      for (const name of SAMPLE_FILES) {
        const resp = await fetch(`/data/samples/${name}`);
        const blob = await resp.blob();
        const fd = new FormData();
        fd.append("files", new File([blob], name, { type: "text/csv" }));
        await api("/datasets", { method: "POST", body: fd });
      }
      await ensureSession();
      await refreshDatasets();
      toast("Sample datasets loaded.");
    } catch (err) {
      toast(err.message, true);
    }
  }

  // ---------------------------------------------------------------- documents
  async function refreshDocuments() {
    try {
      const data = await api("/documents");
      state.documents = data.documents || [];
    } catch (_) {
      state.documents = [];  // document RAG disabled on the server
    }
    renderDocumentList();
  }

  function renderDocumentList() {
    const list = $("#doc-list");
    list.innerHTML = "";
    $("#doc-count").textContent = state.documents.length;
    for (const doc of state.documents) {
      const item = el("div", "doc-item");
      const info = el("div", "doc-name");
      info.appendChild(el("div", "", doc.name));
      info.appendChild(el("div", "doc-meta", `${doc.chunks} chunks${doc.pages ? ` · ${doc.pages} pages` : ""}`));
      const del = el("button", "ds-del", "✕");
      del.title = "Remove document";
      del.addEventListener("click", async () => {
        try {
          await api(`/documents/${doc.doc_id}`, { method: "DELETE" });
          toast(`Removed ${doc.name}`);
          await refreshDocuments();
        } catch (err) { toast(err.message, true); }
      });
      item.appendChild(info);
      item.appendChild(del);
      list.appendChild(item);
    }
  }

  async function uploadDocuments(fileList) {
    if (!fileList || !fileList.length) return;
    const fd = new FormData();
    for (const f of fileList) fd.append("files", f);
    toast(`Indexing ${fileList.length} document(s)…`);
    try {
      const res = await fetch(API + "/documents", { method: "POST", body: fd });
      const data = await res.json();
      const ok = data.imported || [];
      const bad = data.errors || [];
      if (ok.length) toast(`Indexed ${ok.length} document(s): ${ok.reduce((n, d) => n + d.chunks, 0)} chunks.`);
      if (bad.length) toast(`Failed: ${bad[0].error || bad[0].file}`, true);
      await refreshDocuments();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function loadSampleDocuments() {
    try {
      toast("Indexing sample documents…");
      const data = await api("/documents/samples", { method: "POST" });
      await ensureSession();
      await refreshDocuments();
      toast(`Indexed ${data.count} sample documents.`);
    } catch (err) {
      toast(err.message, true);
    }
  }

  function addSources(contentEl, citations) {
    const box = el("div", "sources");
    box.appendChild(el("div", "sources-title", "📚 Sources"));
    const ol = el("ol");
    const seen = new Set();
    for (const c of citations) {
      const key = c.label + (c.sql || "");
      if (seen.has(key)) continue;
      seen.add(key);
      const li = el("li", "", c.label + (c.definition ? ` — ${c.definition}: ` : ""));
      if (c.sql) li.appendChild(el("code", "", c.sql));
      ol.appendChild(li);
    }
    box.appendChild(ol);
    contentEl.appendChild(box);
  }

  // ---------------------------------------------------------------- markdown
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderMarkdown(src) {
    let text = String(src || "").replace(/\r\n/g, "\n");
    // strip trailing whitespace per line
    text = text.replace(/[ \t]+$/gm, "");
    const lines = text.split("\n");
    let html = "";
    let inCode = false;
    let codeBuf = [];
    let listType = null;

    const closeList = () => {
      if (listType) { html += `</${listType}>`; listType = null; }
    };

    for (const line of lines) {
      if (/^```/.test(line)) {
        if (inCode) {
          html += `<pre><button class="copy" onclick="navigator.clipboard.writeText(this.parentNode.querySelector('code').innerText);this.textContent='✓ copied';">copy</button><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`;
          codeBuf = [];
          inCode = false;
        } else {
          closeList();
          inCode = true;
        }
        continue;
      }
      if (inCode) { codeBuf.push(line); continue; }

      let m;
      if ((m = line.match(/^(#{1,4})\s+(.*)/))) {
        closeList();
        const lvl = m[1].length;
        html += `<h${lvl}>${inline(m[2])}</h${lvl}>`;
      } else if (/^\s*[-*]\s+/.test(line)) {
        if (listType !== "ul") { closeList(); listType = "ul"; html += "<ul>"; }
        html += `<li>${inline(line.replace(/^\s*[-*]\s+/, ""))}</li>`;
      } else if (/^\s*\d+\.\s+/.test(line)) {
        if (listType !== "ol") { closeList(); listType = "ol"; html += "<ol>"; }
        html += `<li>${inline(line.replace(/^\s*\d+\.\s+/, ""))}</li>`;
      } else if (/^\s*\|/.test(line)) {
        closeList();
        const cells = line.split("|").slice(1, -1).map((c) => c.trim());
        if (/^[\s:|-]+$/.test(cells.join("|"))) continue; // separator row
        if (!html.endsWith("</table>")) html += "<table>";
        html += "<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
      } else if (/^>\s?/.test(line)) {
        closeList();
        html += `<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`;
      } else if (line.trim() === "") {
        closeList();
        html += "<p></p>";
      } else {
        closeList();
        html += `<p>${inline(line)}</p>`;
      }
    }
    closeList();
    if (inCode) html += `<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`;
    return html;
  }

  function inline(text) {
    let t = escapeHtml(text);
    // images with relative/absolute URL or data uri
    t = t.replace(/!\[([^\]]*)\]\((.*?)\)/g, (_, alt, src) => {
      if (src.startsWith("http") || src.startsWith("/") || src.startsWith("data:")) {
        return `<img class="chart-img" src="${src}" alt="${alt}" loading="lazy"/>`;
      }
      return `[${alt}]`;
    });
    // inline code
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    // bold
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // italic
    t = t.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    // links
    t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return t;
  }

  // ---------------------------------------------------------------- chat
  const messagesEl = $("#messages");

  function appendUserMessage(text) {
    const wrap = el("div", "msg user");
    wrap.appendChild(el("div", "avatar", "🙂"));
    const bubble = el("div", "bubble");
    bubble.innerHTML = `<p>${escapeHtml(text).replace(/\n/g, "<br/>")}</p>`;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollBottom();
    return wrap;
  }

  function appendAssistantStart() {
    const wrap = el("div", "msg assistant");
    wrap.appendChild(el("div", "avatar", "🤖"));
    const body = el("div", "bubble");
    wrap.appendChild(body);
    const thinking = el("div", "thinking");
    thinking.appendChild(el("div", "spinner"));
    thinking.appendChild(document.createTextNode("analysing…"));
    body.appendChild(thinking);
    messagesEl.appendChild(wrap);
    scrollBottom();
    return { wrap, body, content: el("div", "content") };
  }

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage(text) {
    if (!text.trim() || state.streaming) return;
    if (!state.datasets.length && !state.documents.length) {
      toast("Upload a dataset or a document first.", true);
      return;
    }
    await ensureSession();
    const holder = appendUserMessage(text.trim());
    const asst = appendAssistantStart();

    state.streaming = true;
    $("#send").disabled = true;
    $("#input").disabled = true;

    const activeIds = state.datasets.map((d) => d.dataset_id);
    const body = JSON.stringify({ session_id: state.sessionId, message: text.trim(), dataset_ids: activeIds });

    let markdownAcc = "";
    const contentEl = asst.content;
    asst.body.appendChild(contentEl);

    const tools = new Map();
    let finalPayload = null;

    try {
      const res = await fetch(API + "/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          handleEvent(part);
        }
      }

      // process any remaining event
      if (buffer.trim()) handleEvent(buffer);

      function handleEvent(block) {
        let event = "message";
        const dataLines = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) return;
        let data;
        try { data = JSON.parse(dataLines.join("\n")); } catch (_) { return; }

        switch (event) {
          case "note":
            showThinking(asst.body, data.text || "");
            break;
          case "plan":
            showThinking(asst.body, "planning tool calls… " + (data.tools || []).join(", "));
            break;
          case "tool_start":
            showThinking(asst.body, `running tool: ${data.name}…`);
            break;
          case "tool_done":
            tools.set(data.name, { name: data.name, status: "ok" });
            break;
          case "tool_error":
            tools.set(data.name, { name: data.name, status: "err" });
            toast(`Tool ${data.name} failed: ${data.error}`, true);
            break;
          case "token":
            if (asst.body.querySelector(".thinking")) asst.body.querySelector(".thinking").remove();
            markdownAcc += data.text || "";
            contentEl.innerHTML = renderMarkdown(markdownAcc);
            scrollBottom();
            break;
          case "final":
            finalPayload = data.response;
            break;
          case "error":
            throw new Error(data.message);
        }
      }

      // final render
      if (asst.body.querySelector(".thinking")) asst.body.querySelector(".thinking").remove();
      if (finalPayload) {
        let answer = finalPayload.answer || markdownAcc;
        contentEl.innerHTML = renderMarkdown(answer);
        if (finalPayload.route === "documents") {
          contentEl.insertBefore(el("div", "route-badge", "📘 answered from your documents"), contentEl.firstChild);
        }
        if (finalPayload.sql) addCodeBlock(contentEl, "sql", finalPayload.sql);
        if (finalPayload.pandas) addCodeBlock(contentEl, "python", finalPayload.pandas);
        if (finalPayload.citations && finalPayload.citations.length) addSources(contentEl, finalPayload.citations);
        if (finalPayload.reasoning) addReasoning(contentEl, finalPayload.reasoning);
        if (finalPayload.tools && finalPayload.tools.length) addToolsUsed(contentEl, finalPayload.tools);
        if (finalPayload.errors && finalPayload.errors.length) {
          toast(`Some tools failed: ${finalPayload.errors[0]}`, true);
        }
      } else {
        contentEl.innerHTML = renderMarkdown(markdownAcc) || "<p><em>(no answer)</em></p>";
      }
      scrollBottom();
    } catch (err) {
      contentEl.innerHTML = `<p><em style="color:#dc2626">Request failed: ${escapeHtml(err.message)}</em></p>`;
      toast(err.message, true);
    } finally {
      state.streaming = false;
      $("#send").disabled = false;
      $("#input").disabled = false;
      $("#input").focus();
    }
  }

  function showThinking(body, text) {
    let t = body.querySelector(".thinking");
    if (!t) {
      t = el("div", "thinking");
      t.appendChild(el("div", "spinner"));
      t.appendChild(document.createTextNode(""));
      body.appendChild(t);
    }
    t.childNodes[1].nodeValue = text;
    scrollBottom();
  }

  function addCodeBlock(contentEl, lang, code) {
    const pre = el("pre");
    const copy = el("button", "copy", "copy");
    copy.onclick = (e) => { navigator.clipboard.writeText(code); copy.textContent = "✓ copied"; };
    const codeNode = el("code", "", code);
    codeNode.setAttribute("data-lang", lang);
    pre.appendChild(copy);
    pre.appendChild(codeNode);
    contentEl.appendChild(pre);
  }

  function addReasoning(contentEl, text) {
    const det = el("details", "reasoning");
    const summary = el("summary", "", "🧠 Why I answered this way");
    const body = el("div", "r-body", text);
    det.appendChild(summary);
    det.appendChild(body);
    contentEl.appendChild(det);
  }

  function addToolsUsed(contentEl, tools) {
    const row = el("div", "tools-used");
    for (const t of tools) {
      const b = el("span", "tool-badge" + (t.status === "err" ? " err" : ""), `⚙ ${t.name}`);
      row.appendChild(b);
    }
    contentEl.appendChild(row);
  }

  // ---------------------------------------------------------------- dashboard
  async function openDashboard() {
    if (!state.datasets.length) { toast("Upload a dataset first.", true); return; }
    const ids = state.datasets.map((d) => d.dataset_id);
    toast("Building dashboard…");
    try {
      const data = await api("/dashboards", { method: "POST", body: JSON.stringify({ dataset_ids: ids, title: "Data Dashboard" }) });
      const frame = $("#dashboard-frame");
      frame.src = data.url;
      $("#download-dashboard").href = data.download_url;
      $("#dashboard-modal").classList.remove("hidden");
    } catch (err) { toast(err.message, true); }
  }

  async function exportReport() {
    if (!state.pinned) { toast("Select a dataset to export.", true); return; }
    const url = `${API}/reports/${state.pinned}/html`;
    try {
      const res = await fetch(url, { method: "POST" });
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "data-report.html";
      a.click();
      URL.revokeObjectURL(a.href);
      toast("Report downloaded.");
    } catch (err) { toast(err.message, true); }
  }

  // ---------------------------------------------------------------- wire-up
  function init() {
    const input = $("#input");
    const send = $("#send");

    send.addEventListener("click", () => { sendMessage(input.value); input.value = ""; input.style.height = "auto"; });

    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 140) + "px";
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(input.value);
        input.value = "";
        input.style.height = "auto";
      }
    });

    const dz = $("#dropzone");
    const fileInput = $("#file-input");
    dz.addEventListener("click", (e) => { if (e.target.id !== "browse-link") fileInput.click(); });
    $("#browse-link").addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); fileInput.click(); });
    fileInput.addEventListener("change", () => { uploadFiles(fileInput.files); fileInput.value = ""; });
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      uploadFiles(e.dataTransfer.files);
    });

    $("#load-samples").addEventListener("click", loadSamples);
    const docDz = $("#doc-dropzone");
    const docInput = $("#doc-input");
    docDz.addEventListener("click", () => docInput.click());
    docInput.addEventListener("change", () => { uploadDocuments(docInput.files); docInput.value = ""; });
    docDz.addEventListener("dragover", (e) => { e.preventDefault(); docDz.classList.add("dragover"); });
    docDz.addEventListener("dragleave", () => docDz.classList.remove("dragover"));
    docDz.addEventListener("drop", (e) => {
      e.preventDefault();
      docDz.classList.remove("dragover");
      uploadDocuments(e.dataTransfer.files);
    });
    $("#load-sample-docs").addEventListener("click", loadSampleDocuments);
    $("#new-chat-btn").addEventListener("click", async () => {
      resetSession();
      await ensureSession();
      toast("New conversation started.");
    });
    $("#export-report-btn").addEventListener("click", exportReport);

    $("#close-dashboard").addEventListener("click", () => {
      $("#dashboard-modal").classList.add("hidden");
      $("#dashboard-frame").src = "about:blank";
    });
    $("#dashboard-modal").addEventListener("click", (e) => {
      if (e.target.id === "dashboard-modal") $("#dashboard-modal").classList.add("hidden");
    });

    for (const chip of document.querySelectorAll(".chip")) {
      chip.addEventListener("click", () => {
        const q = chip.dataset.q;
        if (/dashboard/i.test(q)) {
          openDashboard();
          return;
        }
        sendMessage(q);
      });
    }

    refreshHealth();
    refreshDatasets();
    refreshDocuments();
    setInterval(refreshHealth, 15000);
    setInterval(() => { if (!state.streaming) refreshDatasets(); }, 20000);
    input.focus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
