const grid = document.getElementById("grid");
const countEl = document.getElementById("count");
const emptyState = document.getElementById("empty-state");
const searchInput = document.getElementById("search");

let allCompanies = [];
let confirmingId = null;

function badge(text, extraClass) {
  const span = document.createElement("span");
  span.className = `badge ${extraClass || ""}`.trim();
  span.textContent = text;
  return span;
}

async function fetchCompanies() {
  const res = await fetch(`${API_BASE}/api/companies`);
  allCompanies = await res.json();
  render();
}

async function deleteCompany(id) {
  await fetch(`${API_BASE}/api/companies/${id}`, { method: "DELETE" });
  allCompanies = allCompanies.filter((c) => c.id !== id);
  confirmingId = null;
  render();
}

function matchesSearch(company, needle) {
  if (!needle) return true;
  const haystack = `${company.name} ${company.website || ""}`.toLowerCase();
  return haystack.includes(needle);
}

function renderCard(company) {
  const card = document.createElement("div");
  card.className = "company-card";

  const h3 = document.createElement("h3");
  h3.textContent = company.name;
  card.appendChild(h3);

  if (company.website) {
    const site = document.createElement("div");
    site.className = "website";
    site.textContent = company.website;
    card.appendChild(site);
  }

  const primarySection = document.createElement("div");
  primarySection.className = "tag-section";
  const primaryLabel = document.createElement("div");
  primaryLabel.className = "tag-label";
  primaryLabel.textContent = "Primary";
  primarySection.appendChild(primaryLabel);
  const primaryRow = document.createElement("div");
  primaryRow.className = "badge-row";
  primaryRow.appendChild(badge(company.primary_tag || "none", "primary"));
  primarySection.appendChild(primaryRow);
  card.appendChild(primarySection);

  const secondarySection = document.createElement("div");
  secondarySection.className = "tag-section";
  const secondaryLabel = document.createElement("div");
  secondaryLabel.className = "tag-label";
  secondaryLabel.textContent = "Secondary";
  secondarySection.appendChild(secondaryLabel);
  const secondaryRow = document.createElement("div");
  secondaryRow.className = "badge-row";
  if (company.secondary_tags && company.secondary_tags.length) {
    company.secondary_tags.forEach((t) => secondaryRow.appendChild(badge(t, "secondary")));
  } else {
    secondaryRow.appendChild(badge("none"));
  }
  secondarySection.appendChild(secondaryRow);
  card.appendChild(secondarySection);

  const kwSection = document.createElement("div");
  kwSection.className = "tag-section";
  const kwLabel = document.createElement("div");
  kwLabel.className = "tag-label";
  kwLabel.textContent = "Keywords";
  kwSection.appendChild(kwLabel);
  const kwLine = document.createElement("div");
  kwLine.className = "keyword-line";
  kwLine.textContent = company.keywords && company.keywords.length ? company.keywords.join(", ") : "none";
  kwSection.appendChild(kwLine);
  card.appendChild(kwSection);

  const footer = document.createElement("div");
  footer.className = "footer-row";

  if (confirmingId === company.id) {
    const confirmRow = document.createElement("div");
    confirmRow.className = "confirm-row";
    confirmRow.style.width = "100%";

    const text = document.createElement("span");
    text.textContent = `Delete ${company.name} permanently?`;
    confirmRow.appendChild(text);

    const yesBtn = document.createElement("button");
    yesBtn.className = "danger";
    yesBtn.textContent = "Yes, delete";
    yesBtn.onclick = () => deleteCompany(company.id);
    confirmRow.appendChild(yesBtn);

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "secondary";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => {
      confirmingId = null;
      render();
    };
    confirmRow.appendChild(cancelBtn);

    card.appendChild(confirmRow);
  } else {
    const delBtn = document.createElement("button");
    delBtn.className = "danger";
    delBtn.textContent = "Delete";
    delBtn.onclick = () => {
      confirmingId = company.id;
      render();
    };
    footer.appendChild(delBtn);
    card.appendChild(footer);
  }

  return card;
}

function render() {
  const needle = searchInput.value.trim().toLowerCase();
  const filtered = allCompanies.filter((c) => matchesSearch(c, needle));

  grid.innerHTML = "";
  filtered.forEach((c) => grid.appendChild(renderCard(c)));

  const noun = filtered.length === 1 ? "company" : "companies";
  countEl.textContent = `${filtered.length} ${noun}`;

  emptyState.style.display = allCompanies.length === 0 ? "block" : "none";
  grid.style.display = allCompanies.length === 0 ? "none" : "grid";
}

searchInput.addEventListener("input", render);

fetchCompanies();
