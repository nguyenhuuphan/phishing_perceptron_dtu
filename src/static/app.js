// app.js — logic phía client cho trang kiểm tra phishing
(function () {
  const form = document.getElementById("check-form");
  const input = document.getElementById("url-input");
  const btn = document.getElementById("check-btn");
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");

  function esc(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function render(data) {
    const isPhish = data.verdict === "PHISHING";
    const badgeCls = isPhish ? "phishing" : "legitimate";
    const pLegit = (data.probability.legitimate * 100).toFixed(2);
    const pPhish = (data.probability.phishing * 100).toFixed(2);

    const topRows = data.top_contributors.length
      ? data.top_contributors.map((t) =>
          `<tr>
             <td>${esc(t.feature)}</td>
             <td>${t.value}</td>
             <td>${t.contribution > 0 ? "+" : ""}${t.contribution}</td>
             <td class="dir-${t.direction}">${t.direction}</td>
           </tr>`).join("")
      : '<tr><td colspan="4" class="muted">Không có đặc trưng nổi bật</td></tr>';

    result.innerHTML = `
      <div class="card">
        <div class="verdict-row">
          <div>
            <div class="muted">URL kiểm tra</div>
            <strong>${esc(data.url)}</strong>
          </div>
          <span class="badge ${badgeCls}">${data.verdict}</span>
        </div>
        <div class="prob-bar">
          <div class="legit" style="width:${pLegit}%"></div>
          <div class="phish" style="width:${pPhish}%"></div>
        </div>
        <div class="grid2">
          <div class="stat"><div class="label">Legitimate</div><div class="value">${pLegit}%</div></div>
          <div class="stat"><div class="label">Phishing</div><div class="value">${pPhish}%</div></div>
          <div class="stat"><div class="label">Biên tin cậy</div><div class="value">${data.margin}</div></div>
          <div class="stat"><div class="label">Đặc trưng trích</div><div class="value">${data.features_count.computed}/${data.features_count.total}</div></div>
        </div>
        ${data.features_count.missing ? `<p class="muted">⚠ ${data.features_count.missing} đặc trưng không trích được (gán 0): ${esc(data.approximated.join(", "))}</p>` : ""}
      </div>

      <div class="card">
        <h3>Đặc trưng đóng góp chính</h3>
        <table>
          <thead><tr><th>Đặc trưng</th><th>Giá trị</th><th>Đóng góp</th><th>Hướng</th></tr></thead>
          <tbody>${topRows}</tbody>
        </table>
      </div>`;
    result.classList.remove("hidden");
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = input.value.trim();
    if (!url) return;

    btn.disabled = true;
    result.classList.add("hidden");
    loading.classList.remove("hidden");

    try {
      const resp = await fetch("/api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Lỗi server");
      render(data);
    } catch (err) {
      result.innerHTML = `<div class="card"><strong style="color:var(--red)">Lỗi:</strong> ${esc(err.message)}</div>`;
      result.classList.remove("hidden");
    } finally {
      loading.classList.add("hidden");
      btn.disabled = false;
    }
  });
})();