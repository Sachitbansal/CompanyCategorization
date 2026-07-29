"""
app.py — Streamlit frontend for optimiseGEO.

Thin UI over main.run_pipeline: consumes the same generator the CLI uses,
so every step it prints to the terminal shows up here too, live, as the
generator yields it (Streamlit renders each st.write() call as it runs
within a single script pass — no polling needed).

The one place this can't just "keep running" is the low-confidence pause
(run_pipeline yields "awaiting_context" and stops instead of calling
input()): that needs a real user click, so it's handled with
st.session_state + a rerun — the pending state (brand name, draft
summary, the already-fetched ScrapedContent so it isn't re-scraped) is
stashed, a form is shown, and submitting it calls run_pipeline again with
extra_context filled in to resume from where it left off.
"""

import streamlit as st

from main import STEP_LABELS, run_pipeline

st.set_page_config(page_title="optimiseGEO", page_icon="🏷️")
st.title("🏷️ optimiseGEO — Brand Categorization")
st.caption("Enter a brand name or a website — which one it is gets detected automatically.")

if "pending" not in st.session_state:
    st.session_state.pending = None


def render_result(result: dict) -> None:
    if result.get("cache_hit"):
        st.info("Found an existing record for this brand — no LLM calls were made.")

    st.subheader(result["name"])
    if result.get("website"):
        st.caption(result["website"])
    st.write(result["summary"])

    st.markdown(f"**Primary tag:** `{result['primary_tag']}`")
    if result["secondary_tags"]:
        st.markdown("**Secondary tags:** " + " ".join(f"`{t}`" for t in result["secondary_tags"]))
    else:
        st.markdown("**Secondary tags:** _none_")


def drive_pipeline(name_or_url: str, extra_context: str | None = None, scraped=None) -> None:
    status = st.status("Running pipeline...", expanded=True)
    result = None
    pending = None

    for event in run_pipeline(name_or_url, extra_context=extra_context, scraped=scraped):
        step = event["step"]
        label = STEP_LABELS.get(step, step)

        if step == "awaiting_context":
            pending = event
        elif step == "complete":
            result = event["result"]
        elif event["status"] == "running":
            status.write(f"⏳ {label}...")
        elif event["status"] == "done":
            detail = event.get("detail")
            status.write(f"✅ {label}" + (f" — {detail}" if detail else ""))
            if step == "retrieve_similar":
                matches = event.get("matches") or []
                if matches:
                    with status.expander("Matched companies (few-shot examples sent to the tagging call)"):
                        for m in matches:
                            st.markdown(
                                f"**{m['name']}** — similarity `{m['similarity']:.2f}`  \n"
                                f"Primary: `{m['primary_tag']}`"
                                + (
                                    "  Secondary: " + ", ".join(f"`{t}`" for t in m["secondary_tags"])
                                    if m["secondary_tags"]
                                    else ""
                                )
                            )
                            st.caption(m["summary"])
                else:
                    status.write("_No company cleared the similarity threshold — tagging call sent with no few-shot examples._")

    if pending is not None:
        status.update(label="Waiting for more context", state="error", expanded=True)
        st.session_state.pending = {
            "name_or_url": name_or_url,
            "brand_name": pending["brand_name"],
            "draft_summary": pending["draft_summary"],
            "scraped": pending["scraped"],
        }
        st.rerun()
    else:
        status.update(label="Done", state="complete", expanded=False)
        st.session_state.pending = None
        render_result(result)


if st.session_state.pending is None:
    with st.form("categorize_form"):
        name_or_url = st.text_input("Brand name or website", placeholder="e.g. Stripe, or stripe.com")
        submitted = st.form_submit_button("Categorize")
    if submitted:
        if name_or_url.strip():
            drive_pipeline(name_or_url.strip())
        else:
            st.warning("Enter a brand name or website first.")

else:
    pending = st.session_state.pending
    st.warning(
        f"Couldn't confidently identify **{pending['brand_name']}** from the site/name alone.\n\n"
        f"Draft summary: _\"{pending['draft_summary']}\"_"
    )
    with st.form("context_form"):
        extra = st.text_area("Describe what this brand does (industry, product, customers)")
        col1, col2 = st.columns(2)
        give_context = col1.form_submit_button("Submit description", use_container_width=True)
        skip = col2.form_submit_button("Keep summary as-is", use_container_width=True)

    if give_context or skip:
        drive_pipeline(
            pending["name_or_url"],
            extra_context=extra.strip() if give_context else "",
            scraped=pending["scraped"],
        )
