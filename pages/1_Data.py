"""
pages/1_Data.py — Streamlit /Data page.

Streamlit auto-discovers anything under pages/ and adds it to the sidebar
nav, so this file alone is what makes the "Data" tab exist next to the
main categorize page (app.py). Lists every company in the DB with its
primary + secondary tags, one card per company, with a delete action for
clearing out test entries (so a brand can be re-categorized from scratch).
"""

import streamlit as st

import db
import embed

st.set_page_config(page_title="optimiseGEO — Data", page_icon="📊", layout="wide")
st.title("📊 Categorized Companies")

conn = db.init_db()

if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None


def delete_company(company_id: int) -> None:
    """Remove the company from both stores: SQLite (row + junction rows +
    category counts) and Chroma (its embedding), so it's fully gone and can
    be categorized fresh next time."""
    embedding_id = db.delete_company(conn, company_id)
    if embedding_id:
        collection = embed.get_chroma_collection()
        embed.delete_embedding(collection, embedding_id)
    st.session_state.confirm_delete_id = None


companies = db.get_all_companies(conn)

if not companies:
    st.info("No companies categorized yet. Go to the main page and categorize one first.")
    st.stop()

search = st.text_input("Filter by name or website", placeholder="e.g. stripe")
if search.strip():
    needle = search.strip().lower()
    companies = [
        c for c in companies
        if needle in (c["name"] or "").lower() or needle in (c["website"] or "").lower()
    ]

st.caption(f"{len(companies)} compan{'y' if len(companies) == 1 else 'ies'}")

COLUMNS = 2
rows = [companies[i:i + COLUMNS] for i in range(0, len(companies), COLUMNS)]

for row in rows:
    cols = st.columns(COLUMNS)
    for col, company in zip(cols, row):
        with col:
            with st.container(border=True):
                primary = db.find_category_by_id(conn, company["primary_tag_id"])
                secondaries = db.get_secondary_tags(conn, company["id"])

                st.markdown(f"### {company['name']}")
                if company["website"]:
                    st.caption(company["website"])

                keywords = company.get("keywords") or []
                if keywords:
                    st.markdown(" ".join(f"`{kw}`" for kw in keywords))
                else:
                    st.caption("_no keywords_")

                if primary:
                    st.markdown(f"🏷️ **Primary:** `{primary['name']}`")
                else:
                    st.markdown("🏷️ **Primary:** _none_")

                if secondaries:
                    badges = " ".join(f"`{tag['name']}`" for tag in secondaries)
                    st.markdown(f"**Secondary:** {badges}")
                else:
                    st.markdown("**Secondary:** _none_")

                if st.session_state.confirm_delete_id == company["id"]:
                    st.warning(f"Delete **{company['name']}** permanently?")
                    yes_col, no_col = st.columns(2)
                    if yes_col.button("Yes, delete", key=f"yes_{company['id']}", use_container_width=True):
                        delete_company(company["id"])
                        st.rerun()
                    if no_col.button("Cancel", key=f"no_{company['id']}", use_container_width=True):
                        st.session_state.confirm_delete_id = None
                        st.rerun()
                else:
                    if st.button("🗑️ Delete", key=f"del_{company['id']}"):
                        st.session_state.confirm_delete_id = company["id"]
                        st.rerun()
