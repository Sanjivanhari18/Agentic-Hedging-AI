(function () {
  const API_BASE = "/api/v1";
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("status");
  const resultsSection = document.getElementById("results-section");
  const errorSection = document.getElementById("error-section");
  const resultMeta = document.getElementById("result-meta");
  const rawTextEl = document.getElementById("raw-text");
  const rawTextWrap = document.getElementById("raw-text-wrap");
  const tableHead = document.getElementById("table-head");
  const tableBody = document.getElementById("table-body");
  const tableWrap = document.getElementById("table-wrap");
  const errorMessage = document.getElementById("error-message");

  const tabs = document.querySelectorAll(".tab");
  let currentMode = "pdf";

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (t) { t.classList.remove("active"); });
      tab.classList.add("active");
      currentMode = tab.getAttribute("data-mode");
      fileInput.accept = currentMode === "pdf" ? ".pdf" : "image/*";
      fileInput.value = "";
    });
  });

  function setStatus(text, type) {
    statusEl.textContent = text;
    statusEl.className = "status" + (type ? " " + type : "");
  }

  function showError(msg) {
    errorSection.hidden = false;
    resultsSection.hidden = true;
    errorMessage.textContent = msg;
  }

  function showResults(data) {
    errorSection.hidden = true;
    resultsSection.hidden = false;

    resultMeta.textContent = "Source: " + (data.source_type || "unknown") +
      " · Pages: " + (data.page_count ?? "—") +
      (data.success ? " · Parsed table: " + (data.columns && data.columns.length ? data.columns.length + " columns" : "—") : "");

    rawTextEl.textContent = data.raw_text || "(no text extracted)";
    rawTextWrap.hidden = !data.raw_text;

    if (data.dataframe && data.dataframe.length > 0 && data.columns && data.columns.length > 0) {
      tableWrap.hidden = false;
      tableHead.innerHTML = "<tr>" + data.columns.map(function (c) { return "<th>" + escapeHtml(String(c)) + "</th>"; }).join("") + "</tr>";
      tableBody.innerHTML = data.dataframe.map(function (row) {
        return "<tr>" + data.columns.map(function (col) {
          const v = row[col];
          return "<td>" + (v == null ? "" : escapeHtml(String(v))) + "</td>";
        }).join("") + "</tr>";
      }).join("");
    } else {
      tableWrap.hidden = false;
      tableHead.innerHTML = "<tr><th>(no columns)</th></tr>";
      tableBody.innerHTML = "<tr><td>No rows parsed</td></tr>";
    }
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("Please select a file.", "error");
      return;
    }

    setStatus("Extracting…", "loading");
    submitBtn.disabled = true;
    errorSection.hidden = true;
    resultsSection.hidden = true;

    const endpoint = currentMode === "pdf" ? API_BASE + "/extract/pdf" : API_BASE + "/extract/image";
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        showError(data.detail || data.error || "Request failed");
        setStatus("Failed", "error");
        return;
      }

      if (data.error && !data.success) {
        showError(data.error);
        setStatus("Failed", "error");
        return;
      }

      showResults(data);
      setStatus("Done.", "success");
    } catch (err) {
      showError(err.message || "Network error");
      setStatus("Failed", "error");
    } finally {
      submitBtn.disabled = false;
    }
  });
})();
