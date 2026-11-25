(() => {
  const query = new URLSearchParams(window.location.search);
  const queryBase = query.get("api");
  let savedBase = "";
  try {
    savedBase = window.localStorage.getItem("mornorkApiBase") || "";
    if (queryBase) window.localStorage.setItem("mornorkApiBase", queryBase);
  } catch { /* Storage can be unavailable in hardened browser contexts. */ }

  const base = (queryBase || savedBase || "http://127.0.0.1:8765").replace(/\/+$/, "");
  window.MORNORK_API_BASE = base;
  window.mornorkApiUrl = path => new URL(path, base + "/").toString();

  let catalogRequest = null;
  let catalogValue = null;
  let catalogLoadedAt = 0;
  window.fetchMornorkModelCatalog = async ({ workspaceId = "w1", force = false } = {}) => {
    if (!force && catalogValue && Date.now() - catalogLoadedAt < 5000) return catalogValue;
    if (!force && catalogRequest) return catalogRequest;
    catalogRequest = (async () => {
      const path = "/api/models/catalog?workspace_id=" + encodeURIComponent(workspaceId);
      const response = await fetch(window.mornorkApiUrl(path), { cache: "no-store" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || ("Catalog request failed (" + response.status + ")"));
      catalogValue = data;
      catalogLoadedAt = Date.now();
      return data;
    })();
    try {
      return await catalogRequest;
    } finally {
      catalogRequest = null;
    }
  };
  window.invalidateMornorkModelCatalog = () => {
    catalogValue = null;
    catalogLoadedAt = 0;
  };
  window.MORNORK_STARTUP_CATALOG = window.fetchMornorkModelCatalog().catch(error => {
    window.MORNORK_STARTUP_CATALOG_ERROR = error;
    return null;
  });
})();
