/* QueryLens — frontend SPA (no build step). */
(() => {
  "use strict";

  const API = "/api";
  const SAMPLE_FILES = ["sales_data.csv", "retail_transactions.csv", "web_traffic.csv"];
  const DOC_EXT = /\.(pdf|md|markdown|txt|docx)$/i;

  const state = {
    sessionId: null,
    datasets: [],
    documents: [],
    pinned: null,
    streaming: false,
    abort: null,
    demo: false,
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const SVG_NS = "http://www.w3.org/2000/svg";
  function icon(name) {
    const svg = document.createElementNS(SVG_NS, "svg");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.appendChild(use);
    return svg;
  }
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } },
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
      throw new Error(res.status === 429 ? detail : `${res.status}: ${detail}`);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("json")) return res.json();
    return res;
  }

  function copyText(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
      const label = btn.querySelector("span");
      btn.querySelector("use").setAttribute("href", "#i-check");
      if (label) label.textContent = "Copied";
      setTimeout(() => {
        btn.querySelector("use").setAttribute("href", "#i-copy");
        if (label) label.textContent = "Copy";
      }, 1500);
    });
  }

  // ---------------------------------------------------------------- sessions
  async function ensureSession() {
    if (state.sessionId) return;
    const data = await api("/chat/session", { method: "POST", body: JSON.stringify({ dataset_ids: state.datasets.map((d) => d.dataset_id) }) });
    state.sessionId = data.session_id;
  }

  async function newChat() {
    if (state.streaming && state.abort) state.abort.abort();
    state.sessionId = null;
    $("#messages").innerHTML = "";
    $("#app").classList.remove("has-messages");
    closeSidebarOnMobile();
    $("#input").focus();
  }

  // ---------------------------------------------------------------- health
  async function refreshHealth() {
    try {
      const h = await api("/health");
      $("#status-dot").className = "dot ok";
      const llm = h.llm;
      const model = llm.model ? llm.model.split("/").pop() : null;
      $("#provider-badge").textContent = model ? `${model} · online` : "Rule engine · online";
      $("#provider-badge").title = llm.model ? `LLM: ${llm.active || llm.provider}/${llm.model}` : "No LLM key: the offline rule engine answers";
      if (h.demo && h.demo.enabled) {
        state.demo = true;
        $("#demo-banner").hidden = false;
        const n = h.demo.rate_limit_per_hour;
        $("#demo-limit").textContent = n ? `Limit: ${n} requests per hour.` : "";
      }
    } catch (e) {
      $("#status-dot").className = "dot err";
      $("#provider-badge").textContent = "offline";
    }
  }

  // ---------------------------------------------------------------- sidebar lists
  function sidebarItem({ iconName, name, meta, active, onClick, onDelete, deleteTitle }) {
    const item = el("div", "sb-item" + (active ? " active" : ""));
    const ic = el("span", "sb-icon");
    ic.appendChild(icon(iconName));
    const text = el("div", "sb-text");
    text.appendChild(el("div", "sb-name", name));
    text.appendChild(el("div", "sb-meta", meta));
    text.title = name;
    item.append(ic, text);
    if (onDelete && !state.demo) {
      const del = el("button", "sb-del");
      del.title = deleteTitle;
      del.setAttribute("aria-label", deleteTitle);
      del.appendChild(icon("x"));
      del.addEventListener("click", (e) => { e.stopPropagation(); onDelete(); });
      item.appendChild(del);
    }
    if (onClick) item.addEventListener("click", onClick);
    return item;
  }

  async function refreshDatasets() {
    const data = await api("/datasets");
    state.datasets = data.datasets;
    if (state.pinned && !state.datasets.find((d) => d.dataset_id === state.pinned)) state.pinned = null;
    if (!state.pinned && state.datasets.length) state.pinned = state.datasets[0].dataset_id;
    renderDatasetList();
    updateSendState();
  }

  function renderDatasetList() {
    const list = $("#dataset-list");
    list.innerHTML = "";
    $("#dataset-count").textContent = state.datasets.length;
    if (!state.datasets.length) list.appendChild(el("div", "sb-empty", "No datasets yet. Drop a CSV anywhere."));
    for (const ds of state.datasets) {
      list.appendChild(sidebarItem({
        iconName: "table",
        name: ds.name,
        meta: `${ds.row_count.toLocaleString()} rows · ${ds.col_count} cols`,
        active: ds.dataset_id === state.pinned,
        deleteTitle: "Remove dataset",
        onDelete: async () => {
          try {
            await api(`/datasets/${ds.dataset_id}`, { method: "DELETE" });
            toast(`Removed ${ds.name}`);
            await refreshDatasets();
          } catch (err) { toast(err.message, true); }
        },
        onClick: async () => {
          state.pinned = ds.dataset_id;
          renderDatasetList();
          closeSidebarOnMobile();
          if (state.sessionId) {
            try { await api(`/chat/session/${state.sessionId}/reset`, { method: "POST" }); } catch (_) { /* ignore */ }
            toast(`Focused on ${ds.name}. Fresh context started.`);
          }
        },
      }));
    }
    const pinned = state.datasets.find((d) => d.dataset_id === state.pinned);
    const pill = $("#active-dataset");
    pill.hidden = !pinned;
    if (pinned) pill.textContent = pinned.name;
  }

  async function refreshDocuments() {
    try {
      const data = await api("/documents");
      state.documents = data.documents || [];
    } catch (_) {
      state.documents = [];  // document RAG disabled on the server
    }
    renderDocumentList();
    updateSendState();
  }

  function renderDocumentList() {
    const list = $("#doc-list");
    list.innerHTML = "";
    $("#doc-count").textContent = state.documents.length;
    if (!state.documents.length) list.appendChild(el("div", "sb-empty", "Add data dictionaries or metric definitions."));
    for (const doc of state.documents) {
      list.appendChild(sidebarItem({
        iconName: "doc",
        name: doc.name,
        meta: `${doc.chunks} chunks${doc.pages ? ` · ${doc.pages} pages` : ""}`,
        deleteTitle: "Remove document",
        onDelete: async () => {
          try {
            await api(`/documents/${doc.doc_id}`, { method: "DELETE" });
            toast(`Removed ${doc.name}`);
            await refreshDocuments();
          } catch (err) { toast(err.message, true); }
        },
      }));
    }
  }

  // ---------------------------------------------------------------- uploads
  function routeFiles(fileList) {
    const files = Array.from(fileList || []);
    const docs = files.filter((f) => DOC_EXT.test(f.name));
    const csvs = files.filter((f) => !DOC_EXT.test(f.name));
    if (csvs.length) uploadFiles(csvs);
    if (docs.length) uploadDocuments(docs);
  }

  async function uploadFiles(files) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    toast(`Uploading ${files.length} file(s)…`);
    try {
      const res = await fetch(API + "/datasets", { method: "POST", body: fd });
      const data = await res.json();
      if (res.status === 429) throw new Error(data.detail);
      const ok = data.imported || [];
      const bad = data.errors || [];
      if (ok.length) {
        await ensureSession();
        state.pinned = ok[ok.length - 1].dataset_id;
        toast(`Imported ${ok.length} dataset(s).`);
      }
      if (bad.length) toast(`Failed: ${bad[0].error || bad[0].file}`, true);
      await refreshDatasets();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function uploadDocuments(files) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    toast(`Indexing ${files.length} document(s)…`);
    try {
      const res = await fetch(API + "/documents", { method: "POST", body: fd });
      const data = await res.json();
      if (res.status === 429) throw new Error(data.detail);
      const ok = data.imported || [];
      const bad = data.errors || [];
      if (ok.length) toast(`Indexed ${ok.length} document(s): ${ok.reduce((n, d) => n + d.chunks, 0)} chunks.`);
      if (bad.length) toast(`Failed: ${bad[0].error || bad[0].file}`, true);
      await refreshDocuments();
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

  // ---------------------------------------------------------------- markdown
  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  const SQL_KW = /^(select|from|where|and|or|not|group|by|order|having|limit|offset|as|on|join|left|right|inner|outer|full|cross|union|all|distinct|case|when|then|else|end|in|is|null|like|ilike|between|with|desc|asc|over|partition|cast|interval|true|false|exists|filter)$/i;
  const PY_KW = /^(import|from|as|def|return|if|else|elif|for|in|not|and|or|is|None|True|False|lambda|with|print)$/;

  function highlight(code, lang) {
    if (lang !== "sql" && lang !== "python") return escapeHtml(code);
    const kw = lang === "sql" ? SQL_KW : PY_KW;
    const comment = lang === "sql" ? "--[^\\n]*" : "#[^\\n]*";
    const re = new RegExp(`(${comment})|('(?:[^'\\\\]|\\\\.)*'|"(?:[^"\\\\]|\\\\.)*")|(\\b\\d+(?:\\.\\d+)?\\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\\s*\\()|([A-Za-z_][A-Za-z0-9_]*)`, "g");
    let out = "";
    let last = 0;
    let m;
    while ((m = re.exec(code))) {
      out += escapeHtml(code.slice(last, m.index));
      const [tok] = m;
      if (m[1]) out += `<span class="tok-com">${escapeHtml(tok)}</span>`;
      // in SQL, double quotes are identifiers ("total"), not strings
      else if (m[2]) out += lang === "sql" && tok[0] === '"' ? escapeHtml(tok) : `<span class="tok-str">${escapeHtml(tok)}</span>`;
      else if (m[3]) out += `<span class="tok-num">${tok}</span>`;
      else if (m[4]) out += kw.test(tok) ? `<span class="tok-kw">${tok}</span>` : `<span class="tok-fn">${tok}</span>`;
      else out += kw.test(tok) ? `<span class="tok-kw">${tok}</span>` : tok;
      last = re.lastIndex;
    }
    return out + escapeHtml(code.slice(last));
  }

  function codeCardHtml(lang, code) {
    const label = { sql: "SQL", python: "Python" }[lang] || lang || "code";
    return `<div class="code-card"><div class="code-head"><span>${escapeHtml(label)}</span>` +
      `<button class="copy-btn" data-copy><svg><use href="#i-copy"/></svg><span>Copy</span></button></div>` +
      `<pre><code>${highlight(code, (lang || "").toLowerCase())}</code></pre></div>`;
  }

  function renderMarkdown(src) {
    const lines = String(src || "").replace(/\r\n/g, "\n").replace(/[ \t]+$/gm, "").split("\n");
    let html = "";
    let code = null;   // { lang, lines } while inside a fence
    let listType = null;
    let table = [];

    const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };
    const flushTable = () => {
      if (!table.length) return;
      const [head, ...rows] = table;
      html += '<div class="table-wrap"><table><thead><tr>' + head.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table></div>";
      table = [];
    };

    for (const line of lines) {
      const fence = line.match(/^```\s*(\w*)/);
      if (fence) {
        if (code) { html += codeCardHtml(code.lang, code.lines.join("\n")); code = null; }
        else { closeList(); flushTable(); code = { lang: fence[1], lines: [] }; }
        continue;
      }
      if (code) { code.lines.push(line); continue; }

      if (/^\s*\|/.test(line)) {
        closeList();
        const cells = line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
        if (!/^[\s:|-]+$/.test(cells.join("|"))) table.push(cells);
        continue;
      }
      flushTable();
      let m;
      if ((m = line.match(/^(#{1,4})\s+(.*)/))) {
        closeList();
        html += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`;
      } else if (/^\s*[-*]\s+/.test(line)) {
        if (listType !== "ul") { closeList(); listType = "ul"; html += "<ul>"; }
        html += `<li>${inline(line.replace(/^\s*[-*]\s+/, ""))}</li>`;
      } else if (/^\s*\d+\.\s+/.test(line)) {
        if (listType !== "ol") { closeList(); listType = "ol"; html += "<ol>"; }
        html += `<li>${inline(line.replace(/^\s*\d+\.\s+/, ""))}</li>`;
      } else if (/^>\s?/.test(line)) {
        closeList();
        html += `<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`;
      } else if (line.trim() === "") {
        closeList();
      } else {
        closeList();
        html += `<p>${inline(line)}</p>`;
      }
    }
    closeList();
    flushTable();
    if (code) html += codeCardHtml(code.lang, code.lines.join("\n"));
    return html;
  }

  function inline(text) {
    let t = escapeHtml(text);
    t = t.replace(/!\[([^\]]*)\]\((.*?)\)/g, (_, alt, src) => {
      if (src.startsWith("http") || src.startsWith("/") || src.startsWith("data:")) {
        return `<img class="chart-img" src="${src}" alt="${alt}" loading="lazy"/>`;
      }
      return `[${alt}]`;
    });
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return t;
  }

  // ---------------------------------------------------------------- chat
  const threadEl = $("#thread");
  const messagesEl = $("#messages");

  function nearBottom() {
    return threadEl.scrollHeight - threadEl.scrollTop - threadEl.clientHeight < 120;
  }
  function scrollBottom(force = false) {
    if (force || nearBottom()) threadEl.scrollTop = threadEl.scrollHeight;
  }

  function appendUserMessage(text) {
    $("#app").classList.add("has-messages");
    const wrap = el("div", "msg user");
    wrap.appendChild(el("div", "bubble", text));
    messagesEl.appendChild(wrap);
    scrollBottom(true);
  }

  function appendAssistant() {
    const wrap = el("div", "msg assistant");
    const avatar = el("div", "avatar");
    avatar.appendChild(icon("logo"));
    const body = el("div", "body");
    const content = el("div", "content");
    body.appendChild(content);
    wrap.append(avatar, body);
    messagesEl.appendChild(wrap);
    setThinking(content, "Thinking");
    scrollBottom(true);
    return { body, content };
  }

  function setThinking(content, text) {
    let t = content.querySelector(".thinking");
    if (!t) {
      t = el("div", "thinking");
      const pulse = el("span", "pulse");
      pulse.innerHTML = "<i></i><i></i><i></i>";
      t.append(pulse, el("span", "thinking-text"));
      content.innerHTML = "";
      content.appendChild(t);
    }
    t.querySelector(".thinking-text").textContent = text;
  }

  function prettyTool(name) {
    return String(name || "").replace(/_/g, " ");
  }

  async function sendMessage(text) {
    text = (text || "").trim();
    if (!text || state.streaming) return;
    if (!state.datasets.length && !state.documents.length) {
      toast("Upload a dataset or a document first.", true);
      return;
    }
    try { await ensureSession(); } catch (err) { toast(err.message, true); return; }
    appendUserMessage(text);
    const asst = appendAssistant();
    const contentEl = asst.content;

    state.streaming = true;
    state.abort = new AbortController();
    updateSendState();

    let markdownAcc = "";
    let finalPayload = null;

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
          if (!markdownAcc) setThinking(contentEl, data.text || "Working");
          break;
        case "plan":
          if (!markdownAcc) setThinking(contentEl, "Planning: " + (data.tools || []).map(prettyTool).join(", "));
          break;
        case "tool_start":
          if (!markdownAcc) setThinking(contentEl, `Running ${prettyTool(data.name)}`);
          break;
        case "tool_error":
          toast(`Tool ${data.name} failed: ${data.error}`, true);
          break;
        case "token":
          markdownAcc += data.text || "";
          contentEl.innerHTML = renderMarkdown(markdownAcc);
          contentEl.classList.add("cursor");
          scrollBottom();
          break;
        case "final":
          finalPayload = data.response;
          break;
        case "error":
          throw new Error(data.message);
      }
    }

    try {
      const res = await fetch(API + "/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, message: text, dataset_ids: state.datasets.map((d) => d.dataset_id) }),
        signal: state.abort.signal,
      });
      if (!res.ok) {
        let detail = `Server responded ${res.status}`;
        try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
        throw new Error(detail);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) handleEvent(part);
      }
      if (buffer.trim()) handleEvent(buffer);
      renderFinal(asst, finalPayload, markdownAcc);
    } catch (err) {
      contentEl.classList.remove("cursor");
      if (err.name === "AbortError") {
        if (!markdownAcc) contentEl.innerHTML = "";
        contentEl.appendChild(el("p", "error-note", "Stopped."));
      } else {
        contentEl.innerHTML = "";
        contentEl.appendChild(el("p", "error-note", err.message));
      }
    } finally {
      state.streaming = false;
      state.abort = null;
      updateSendState();
      $("#input").focus();
    }
  }

  function renderFinal(asst, payload, streamed) {
    const contentEl = asst.content;
    contentEl.classList.remove("cursor");
    const answer = (payload && payload.answer) || streamed;
    contentEl.innerHTML = renderMarkdown(answer) || '<p class="error-note">(no answer)</p>';
    if (!payload) return;

    if (payload.route === "documents") {
      const badge = el("div", "route-badge");
      badge.append(icon("book"), document.createTextNode("Answered from your documents"));
      contentEl.insertBefore(badge, contentEl.firstChild);
    }
    if (payload.sql) contentEl.insertAdjacentHTML("beforeend", codeCardHtml("sql", payload.sql));
    if (payload.pandas) addDetails(contentEl, "Pandas code", null, codeCardHtml("python", payload.pandas));
    if (payload.citations && payload.citations.length) addSources(contentEl, payload.citations);
    if (payload.reasoning) addDetails(contentEl, "How this was answered", payload.reasoning);
    if (payload.errors && payload.errors.length) toast(`Some tools failed: ${payload.errors[0]}`, true);

    const actions = el("div", "msg-actions");
    const copy = el("button", "icon-btn");
    copy.title = "Copy answer";
    copy.setAttribute("aria-label", "Copy answer");
    copy.appendChild(icon("copy"));
    copy.addEventListener("click", () => copyText(copy, answer));
    actions.appendChild(copy);
    if (payload.tools && payload.tools.length) {
      const tags = el("div", "tool-tags");
      for (const t of payload.tools) tags.appendChild(el("span", "tool-tag" + (t.status === "err" ? " err" : ""), t.name));
      actions.appendChild(tags);
    }
    asst.body.appendChild(actions);
    scrollBottom();
  }

  function addSources(contentEl, citations) {
    const box = el("div", "sources");
    box.appendChild(el("div", "sources-title", "Sources"));
    const list = el("div", "source-list");
    const seen = new Set();
    let n = 0;
    for (const c of citations) {
      const key = c.label + (c.sql || "");
      if (seen.has(key)) continue;
      seen.add(key);
      const item = el("div", "source");
      item.appendChild(el("span", "source-n", String(++n)));
      const text = el("div");
      text.appendChild(document.createTextNode(c.label + (c.definition ? ` · ${c.definition}` : "")));
      if (c.sql) {
        text.appendChild(document.createElement("br"));
        text.appendChild(el("code", "", c.sql));
      }
      item.appendChild(text);
      list.appendChild(item);
    }
    box.appendChild(list);
    contentEl.appendChild(box);
  }

  function addDetails(contentEl, title, text, html) {
    const det = el("details", "details");
    const summary = el("summary");
    summary.append(icon("chevron"), document.createTextNode(title));
    det.appendChild(summary);
    if (text) det.appendChild(el("div", "d-body", text));
    if (html) det.insertAdjacentHTML("beforeend", html);
    contentEl.appendChild(det);
  }

  // ---------------------------------------------------------------- dashboard / report
  async function openDashboard() {
    if (!state.datasets.length) { toast("Upload a dataset first.", true); return; }
    toast("Building dashboard…");
    try {
      const data = await api("/dashboards", { method: "POST", body: JSON.stringify({ dataset_ids: state.datasets.map((d) => d.dataset_id), title: "Data Dashboard" }) });
      $("#dashboard-frame").src = data.url;
      $("#download-dashboard").href = data.download_url;
      $("#dashboard-modal").classList.remove("hidden");
    } catch (err) { toast(err.message, true); }
  }

  function closeDashboard() {
    $("#dashboard-modal").classList.add("hidden");
    $("#dashboard-frame").src = "about:blank";
  }

  async function exportReport() {
    if (!state.pinned) { toast("Select a dataset to export.", true); return; }
    try {
      const res = await fetch(`${API}/reports/${state.pinned}/html`, { method: "POST" });
      if (!res.ok) throw new Error(`Export failed (${res.status})`);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "data-report.html";
      a.click();
      URL.revokeObjectURL(a.href);
      toast("Report downloaded.");
    } catch (err) { toast(err.message, true); }
  }

  // ---------------------------------------------------------------- layout: theme, sidebar, composer
  const mobile = () => window.matchMedia("(max-width: 820px)").matches;

  function effectiveTheme() {
    const set = document.documentElement.dataset.theme;
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function syncThemeIcon() {
    $("#theme-toggle use").setAttribute("href", effectiveTheme() === "dark" ? "#i-sun" : "#i-moon");
  }
  function toggleTheme() {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("ql-theme", next);
    syncThemeIcon();
  }

  function openSidebar() {
    if (mobile()) $("#app").classList.add("sb-open");
    else { $("#app").classList.remove("sb-collapsed"); store.set("ql-sidebar", "open"); }
  }
  function closeSidebar() {
    if (mobile()) $("#app").classList.remove("sb-open");
    else { $("#app").classList.add("sb-collapsed"); store.set("ql-sidebar", "closed"); }
  }
  function closeSidebarOnMobile() { if (mobile()) $("#app").classList.remove("sb-open"); }

  function updateSendState() {
    const send = $("#send");
    const hasText = $("#input").value.trim().length > 0;
    send.classList.toggle("stop", state.streaming);
    send.querySelector("use").setAttribute("href", state.streaming ? "#i-stop" : "#i-send");
    send.title = state.streaming ? "Stop" : "Send";
    send.disabled = !state.streaming && !hasText;
  }

  function autosize() {
    const input = $("#input");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  function submit() {
    if (state.streaming) { state.abort && state.abort.abort(); return; }
    const input = $("#input");
    const text = input.value;
    if (!text.trim()) return;
    input.value = "";
    autosize();
    updateSendState();
    sendMessage(text);
  }

  // ---------------------------------------------------------------- wire-up
  function init() {
    const input = $("#input");
    $("#composer").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
    input.addEventListener("input", () => { autosize(); updateSendState(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!state.streaming) submit(); }
    });

    // files: attach button, sidebar "+" buttons, drag-and-drop anywhere
    const fileInput = $("#file-input");
    const pick = (accept) => { fileInput.accept = accept; fileInput.click(); };
    $("#attach-btn").addEventListener("click", () => pick(".csv,text/csv,.pdf,.md,.markdown,.txt,.docx"));
    $("#add-dataset").addEventListener("click", () => pick(".csv,text/csv"));
    $("#add-document").addEventListener("click", () => pick(".pdf,.md,.markdown,.txt,.docx"));
    fileInput.addEventListener("change", () => { routeFiles(fileInput.files); fileInput.value = ""; });
    let dragDepth = 0;
    const overlay = $("#drop-overlay");
    window.addEventListener("dragenter", (e) => {
      if (!e.dataTransfer || !Array.from(e.dataTransfer.types).includes("Files")) return;
      dragDepth++;
      overlay.classList.add("show");
    });
    window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; overlay.classList.remove("show"); } });
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      dragDepth = 0;
      overlay.classList.remove("show");
      if (e.dataTransfer && e.dataTransfer.files.length) routeFiles(e.dataTransfer.files);
    });

    $("#load-samples").addEventListener("click", loadSamples);
    $("#load-sample-docs").addEventListener("click", loadSampleDocuments);
    $("#new-chat-btn").addEventListener("click", newChat);
    $("#new-chat-top").addEventListener("click", newChat);
    $("#export-report-btn").addEventListener("click", exportReport);
    $("#dashboard-btn").addEventListener("click", openDashboard);
    $("#close-dashboard").addEventListener("click", closeDashboard);
    $("#dashboard-modal").addEventListener("click", (e) => { if (e.target.id === "dashboard-modal") closeDashboard(); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { closeDashboard(); closeSidebarOnMobile(); }
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
    });

    $("#sidebar-open").addEventListener("click", openSidebar);
    $("#sidebar-close").addEventListener("click", closeSidebar);
    $("#scrim").addEventListener("click", closeSidebarOnMobile);
    if (store.get("ql-sidebar") === "closed" && !mobile()) $("#app").classList.add("sb-collapsed");

    $("#theme-toggle").addEventListener("click", toggleTheme);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeIcon);
    syncThemeIcon();

    for (const b of document.querySelectorAll("[data-q], [data-action]")) {
      b.addEventListener("click", () => (b.dataset.action === "dashboard" ? openDashboard() : sendMessage(b.dataset.q)));
    }
    // copy buttons inside rendered code cards
    messagesEl.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-copy]");
      if (btn) copyText(btn, btn.closest(".code-card").querySelector("code").innerText);
    });

    refreshHealth().then(() => { refreshDatasets(); refreshDocuments(); });
    setInterval(refreshHealth, 15000);
    setInterval(() => { if (!state.streaming) refreshDatasets().catch(() => {}); }, 20000);
    input.focus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
