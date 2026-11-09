"""Responsive customer portal for ShopEase Support AI."""
from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from frontend.presentation import (
    customer_status,
    customer_ticket_status,
    customer_timeline,
    pipeline_stage_cards,
)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_URL = os.getenv("SHOPEASE_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "local-demo-key-change-me")
REQUEST_TIMEOUT_SECONDS = 60
APP_NAME = "ShopEase Support AI"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def api_request(
    method: str,
    path: str,
    *,
    customer_id: str | None = None,
    staff_key: str | None = None,
    **kwargs: Any,
) -> tuple[Any | None, str | None]:
    """Call the backend and return explicit API errors for the UI to display."""
    headers = dict(kwargs.pop("headers", {}))
    headers["X-Demo-Api-Key"] = DEMO_API_KEY
    if customer_id:
        headers["X-Customer-Id"] = customer_id
    if staff_key:
        headers["X-Human-Review-Key"] = staff_key
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        return None, f"Could not reach the support service at {API_URL}: {exc}"

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        return None, f"Support service returned HTTP {response.status_code}: {detail}"
    if response.status_code == 204:
        return None, None
    if "text/csv" in response.headers.get("content-type", ""):
        return response.content, None
    try:
        return response.json(), None
    except ValueError:
        return None, "Support service returned an unexpected response."


def show_api_error(message: str | None) -> None:
    if message:
        st.error(message)
        st.caption("Check that the backend is running and that its API configuration is valid.")


def login_page() -> None:
    with st.sidebar:
        st.markdown(f"## 🛍️ {APP_NAME}")
        st.caption("CUSTOMER SUPPORT")
        if st.button("Open staff review console", width="stretch"):
            st.session_state["staff_portal"] = True
            st.rerun()

    customers, error = api_request("GET", "/demo/customers")
    if error:
        show_api_error(error)
        return
    if not isinstance(customers, list) or not customers:
        st.error("No demo customer accounts are available. Seed the demo database and restart the API.")
        return

    customer_options = {
        f"{customer['name']} - {customer['email']} ({customer['customer_id']})": customer["customer_id"]
        for customer in customers
        if all(customer.get(field) for field in ("name", "email", "customer_id"))
    }
    if not customer_options:
        st.error("The service returned no valid demo customer accounts.")
        return

    spacer, form_col = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    with spacer:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#39227e 0%,#6844db 58%,#159caa 100%);border-radius:22px;color:#fff;padding:2.4rem;min-height:360px;box-shadow:0 14px 32px rgba(75,55,169,.2)">
              <div style="color:#e8e4ff;font-size:.76rem;font-weight:750;letter-spacing:.14em">{APP_NAME.upper()}</div>
              <h1 style="color:#fff;margin-top:1rem">A little help goes a long way.</h1>
              <p style="color:#fff;line-height:1.7">Sign in to ask a question, check on your support requests, and get help with your orders.</p>
              <ul>
                <li style="color:#fff;line-height:1.7">Support questions linked to your selected demo account</li>
                <li style="color:#fff;line-height:1.7">Private ticket history scoped to that account</li>
                <li style="color:#fff;line-height:1.7">Clear status and next steps for every request</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with form_col:
        with st.container(border=True):
            st.markdown(
                '<div style="color:#6844db;font-size:.76rem;font-weight:750;letter-spacing:.12em">WELCOME BACK</div>',
                unsafe_allow_html=True,
            )
            st.title("Sign in")
            st.caption("Choose one of the customer accounts seeded for this local demo.")
            with st.form("demo_sign_in"):
                selected = st.selectbox("Demo customer account", list(customer_options))
                submitted = st.form_submit_button("Continue to support", type="primary", width="stretch")
            st.markdown(
                '<div style="background:#fff8e6;border:1px solid #f0d998;border-radius:10px;color:#614b10;padding:.8rem 1rem;font-size:.9rem"><strong>Demo sign-in:</strong> this project currently has no '
                'password-based customer authentication. The selected sample account is not proof of '
                'identity and must not be used for production.</div>',
                unsafe_allow_html=True,
            )
            if submitted:
                customer_id = customer_options[selected]
                user, error = api_request("GET", "/me", customer_id=customer_id)
                if error:
                    show_api_error(error)
                elif not isinstance(user, dict) or user.get("customer_id") != customer_id:
                    st.error("The service could not verify this demo account.")
                else:
                    st.session_state["customer"] = user
                    st.session_state.pop("latest_ticket", None)
                    st.session_state.pop("pending_support_request", None)
                    st.session_state.pop("support_request_error", None)
                    st.rerun()


def render_sidebar(customer: dict[str, Any]) -> str:
    with st.sidebar:
        st.markdown(f"## 🛍️ {APP_NAME}")
        st.caption("CUSTOMER SUPPORT")
        st.divider()
        st.markdown("**Signed in to demo**")
        st.write(customer.get("name", "Customer"))
        st.caption(customer.get("email", ""))
        st.caption(f"Account · {customer.get('customer_id', '')}")
        page = st.radio("Go to", ["Ask support", "My tickets", "Settings"], label_visibility="collapsed")
        st.divider()
        if st.button("Staff review console", width="stretch"):
            st.session_state["staff_portal"] = True
            st.rerun()
        st.caption("LOCAL DEMO MODE")
        if st.button("Sign out", width="stretch"):
            st.session_state.pop("customer", None)
            st.session_state.pop("latest_ticket", None)
            st.session_state.pop("pending_support_request", None)
            st.session_state.pop("support_request_error", None)
            st.rerun()
        return page


def render_ticket_history(customer_id: str) -> None:
    st.title("My support requests")
    st.caption("Only tickets associated with the signed-in demo account are shown.")
    tickets, error = api_request("GET", "/support/tickets", customer_id=customer_id)
    if error:
        show_api_error(error)
        return
    if not isinstance(tickets, list):
        st.error("The service returned an invalid ticket list.")
        return
    if not tickets:
        st.info("No support requests yet. Ask us a question to get started.")
        return

    total, resolved, escalated = st.columns(3)
    total.metric("Total requests", len(tickets))
    resolved.metric("Resolved", sum(ticket.get("status") == "Resolved" for ticket in tickets))
    escalated.metric("Needs follow-up", sum(ticket.get("status") == "Escalated" for ticket in tickets))

    frame = pd.DataFrame(
        [
            {
                "Ticket": ticket["ticket_id"],
                "Date": ticket.get("created_at", "")[:16].replace("T", " "),
                "Order": ticket.get("order_id") or "—",
                "Category": ticket.get("category", ""),
                "Urgency": ticket.get("urgency", ""),
                "Status": customer_ticket_status(
                    ticket.get("status", "Unknown"),
                    ticket.get("human_review_status"),
                ),
            }
            for ticket in tickets
        ]
    )
    export_frame = pd.DataFrame(
        [
            {
                "ticket_id": ticket["ticket_id"],
                "timestamp": ticket.get("created_at", ""),
                "customer_name": ticket.get("customer_name", ""),
                "order_id": ticket.get("order_id") or "",
                "complaint": ticket.get("complaint", ""),
                "category": ticket.get("category", ""),
                "urgency": ticket.get("urgency", ""),
                "sentiment": ticket.get("sentiment", ""),
                "kb_sources": ", ".join(ticket.get("policy_sources", [])),
                "response": ticket.get("response", ""),
                "status": ticket.get("status", ""),
                "human_review_status": ticket.get("human_review_status") or "",
                "human_reviewed_at": ticket.get("human_reviewed_at") or "",
                "escalation_reason": ticket.get("reason") or "",
            }
            for ticket in tickets
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        "Download my ticket history",
        data=export_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="my_shopease_tickets.csv",
        mime="text/csv",
    )
    st.subheader("Request details")
    selected_id = st.selectbox("Choose a ticket", [ticket["ticket_id"] for ticket in tickets])
    selected = next(ticket for ticket in tickets if ticket["ticket_id"] == selected_id)
    with st.container(border=True):
        st.markdown(
            f"**{selected['ticket_id']}** · "
            f"{customer_ticket_status(selected.get('status', 'Unknown'), selected.get('human_review_status'))}"
        )
        if selected.get("human_review_status") == "Approved":
            st.info(
                "The staff review was approved and recorded. This does not mean a refund, "
                "order change, or customer notification has been completed."
            )
        elif selected.get("human_review_status") == "Rejected":
            st.warning(
                "The staff review decision was rejected. The request remains marked for "
                "follow-up; this decision does not change the order or issue a refund."
            )
        st.write(selected.get("complaint", ""))
        st.caption(
            f"Category: {selected.get('category', 'Other')} · "
            f"Urgency: {selected.get('urgency', 'Unknown')}"
        )
        sources = selected.get("policy_sources", [])
        if sources:
            st.caption(f"Guidance used: {', '.join(sources)}")
        st.markdown("**Support response**")
        st.write(selected.get("response", ""))
        with st.expander("What happened with your request"):
            for step in customer_timeline(selected.get("timeline", [])):
                st.markdown(f"**{step['title']}**")
                st.caption(step["detail"])


def render_support_page(customer: dict[str, Any]) -> None:
    customer_id = customer["customer_id"]
    health, health_error = api_request("GET", "/health", customer_id=customer_id)
    if health_error:
        show_api_error(health_error)
    support_mode = health.get("support_pipeline") if isinstance(health, dict) else "demo"
    mode_label = "AUTOGEN GRAPHFLOW" if support_mode == "graphflow" else "LOCAL DEMO"
    deepeval_enabled = bool(health.get("deepeval_enabled")) if isinstance(health, dict) else False
    deepeval_status = health.get("deepeval_status", "disabled") if isinstance(health, dict) else "disabled"
    orders, orders_error = api_request("GET", "/me/orders", customer_id=customer_id)
    tickets, tickets_error = api_request("GET", "/support/tickets", customer_id=customer_id)
    if tickets_error:
        show_api_error(tickets_error)
        tickets = []
    elif not isinstance(tickets, list):
        st.error("The service returned an invalid ticket list.")
        tickets = []

    first_name = escape(str(customer.get("name", "there")).split()[0])
    st.markdown(
        f"""
        <div style="background:linear-gradient(112deg,#39227e 0%,#6844db 53%,#159caa 100%);border-radius:22px;color:#fff;padding:1.55rem 1.8rem;margin:.25rem 0 1.1rem;box-shadow:0 14px 32px rgba(75,55,169,.2)">
          <div style="color:#e8e4ff;font-size:.72rem;letter-spacing:.14em;font-weight:750">CUSTOMER CARE · {escape(mode_label)}</div>
          <h2 style="color:#fff;margin:.35rem 0 .2rem;font-size:1.55rem">Welcome back, {first_name}</h2>
          <p style="color:#fff;margin:0;opacity:.9">Ask a question, follow each step, and see when a person needs to help.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    total, resolved, follow_up, active_orders = st.columns(4)
    total.metric("Support requests", len(tickets))
    resolved.metric("Answered", sum(ticket.get("status") == "Resolved" for ticket in tickets))
    follow_up.metric("Needs a person", sum(ticket.get("status") == "Escalated" for ticket in tickets))
    if orders_error:
        show_api_error(orders_error)
        active_orders.metric("Your orders", "Unavailable")
        order_options = ["No order selected"]
    elif isinstance(orders, list):
        valid_orders = [
            order["order_id"]
            for order in orders
            if isinstance(order, dict) and order.get("order_id")
        ]
        active_orders.metric("Your orders", len(valid_orders))
        order_options = ["No order selected"] + valid_orders
    else:
        st.error("The service returned an invalid order list.")
        active_orders.metric("Your orders", 0)
        order_options = ["No order selected"]

    latest = st.session_state.get("latest_ticket")
    latest_for_customer = (
        latest if isinstance(latest, dict) and latest.get("customer_id") == customer_id else None
    )
    if latest_for_customer and not tickets_error and isinstance(tickets, list):
        matching_ticket = next(
            (
                ticket for ticket in tickets
                if isinstance(ticket, dict)
                and ticket.get("ticket_id") == latest_for_customer.get("ticket_id")
            ),
            None,
        )
        if matching_ticket:
            latest_for_customer = {
                **latest_for_customer,
                "human_review_status": matching_ticket.get("human_review_status"),
                "human_reviewed_at": matching_ticket.get("human_reviewed_at"),
            }
        else:
            st.session_state.pop("latest_ticket", None)
            latest_for_customer = None
    pending_request = st.session_state.get("pending_support_request")
    request_in_progress = (
        isinstance(pending_request, dict)
        and pending_request.get("customer_id") == customer_id
    )
    with st.container():
        st.subheader("How your request moves")
        if request_in_progress:
            st.info("A new request has started. The stages are reset for this request.")
        else:
            st.caption(
                "The numbered stages below show the real path used by this service. "
                "Conditional checks are marked when they do not apply."
            )
        stages = pipeline_stage_cards(
            str(support_mode or "demo"),
            deepeval_enabled,
            latest_for_customer.get("timeline", []) if latest_for_customer else [],
            deepeval_status=str(deepeval_status),
            ticket_status=latest_for_customer.get("status") if latest_for_customer else None,
            human_review_status=(
                latest_for_customer.get("human_review_status")
                if latest_for_customer else None
            ),
            in_progress=request_in_progress,
        )
        stage_markup = []
        for index, stage in enumerate(stages, start=1):
            state_class = str(stage["state"]).casefold().replace(" ", "-")
            warning_state = state_class in {
                "pending", "rejected", "check-failed", "unavailable", "key-required", "no-evidence"
            }
            muted_state = state_class in {"off", "not-needed", "ready", "not-run"}
            status_background, status_color = (
                ("#fff0e8", "#a64b22") if warning_state
                else ("#eee9ff", "#5738c5") if state_class == "in-progress"
                else ("#f0eff7", "#65647e") if muted_state
                else ("#e8f7f2", "#16704f")
            )
            stage_markup.append(
                '<div style="background:#fff;border:1px solid #dcdcf0;border-top:3px solid #6844db;'
                'border-radius:14px;box-shadow:0 6px 18px rgba(38,35,83,.07);'
                'flex:1 0 170px;min-height:145px;padding:.85rem .9rem">'
                '<div style="display:flex;align-items:center;justify-content:space-between;gap:.3rem">'
                f'<span style="display:inline-flex;align-items:center;justify-content:center;width:1.55rem;height:1.55rem;'
                f'border-radius:999px;background:#eee9ff;color:#5738c5;font-size:.68rem;font-weight:800">{index:02d}</span>'
                f'<span style="background:{status_background};color:{status_color};border-radius:999px;'
                f'font-size:.61rem;font-weight:800;letter-spacing:.04em;padding:.25rem .4rem;white-space:nowrap">'
                f'{escape(stage["state"])}</span>'
                '</div>'
                f'<h4 style="color:#20233f;font-size:.87rem;line-height:1.3;margin:.7rem 0 .32rem">{escape(stage["title"])}</h4>'
                f'<p style="color:#626985;font-size:.73rem;line-height:1.45;margin:0">{escape(stage["detail"])}</p>'
                '</div>'
            )
        st.markdown(
            '<div style="display:flex;gap:.65rem;overflow-x:auto;padding:.15rem .1rem .75rem">'
            f'{"".join(stage_markup)}</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Start a conversation")
            st.caption("Share the issue in your own words. Add an order if it is relevant.")
            with st.form("support_question", clear_on_submit=True):
                selected_order = st.selectbox("Your order (optional)", order_options)
                st.caption("Only orders belonging to this signed-in account can be accessed.")
                complaint = st.text_area(
                    "Your question",
                    placeholder="Tell us what you need help with…",
                    height=140,
                    max_chars=4000,
                )
                submitted = st.form_submit_button("Get support", type="primary", width="stretch")
            st.caption("Please don’t include passwords, payment details, or other sensitive information.")
            if submitted:
                if len(complaint.strip()) < 3:
                    st.warning("Please enter a little more detail so support can help.")
                else:
                    order_id = selected_order if selected_order != "No order selected" else None
                    payload = {"complaint": complaint.strip(), "order_id": order_id}
                    if latest_for_customer:
                        st.session_state.pop("latest_ticket", None)
                    st.session_state["pending_support_request"] = {
                        "customer_id": customer_id,
                        "payload": payload,
                    }
                    st.session_state.pop("support_request_error", None)
                    st.rerun()

            request_error = st.session_state.pop("support_request_error", None)
            if request_error:
                show_api_error(request_error)

            pending_request = st.session_state.get("pending_support_request")
            if (
                isinstance(pending_request, dict)
                and pending_request.get("customer_id") == customer_id
            ):
                with st.spinner("Sending your request…"):
                    result, error = api_request(
                        "POST",
                        "/support/tickets",
                        customer_id=customer_id,
                        json=pending_request["payload"],
                    )
                st.session_state.pop("pending_support_request", None)
                if error:
                    st.session_state["support_request_error"] = error
                    st.rerun()
                if isinstance(result, dict):
                    st.session_state["latest_ticket"] = result
                    st.session_state.pop("support_request_error", None)
                    st.rerun()
                st.session_state["support_request_error"] = (
                    "The service did not return a support ticket."
                )
                st.rerun()

    with right:
        st.subheader("Your support conversation")
        if latest_for_customer:
            with st.container(border=True):
                status = latest_for_customer.get("status")
                review_status = latest_for_customer.get("human_review_status")
                if status == "Resolved":
                    st.success("We have an answer")
                elif review_status == "Approved":
                    st.info(
                        "A staff member approved the review decision. The request remains "
                        "in follow-up; approval does not issue a refund or change an order."
                    )
                elif review_status == "Rejected":
                    st.warning(
                        "A staff member rejected the review decision. The request remains "
                        "in follow-up."
                    )
                elif status == "Escalated":
                    st.warning("A support specialist needs to review this")
                else:
                    st.warning(customer_status(status or "Status unavailable"))
                st.caption(f"Request {latest_for_customer.get('ticket_id', '')}")
                st.write(latest_for_customer.get("response", "No response was returned."))
                with st.expander("See what happened at each step", expanded=True):
                    for step in customer_timeline(latest_for_customer.get("timeline", [])):
                        st.markdown(
                            '<div style="border-left:2px solid #dcd4fa;margin-left:.35rem;'
                            'padding:.15rem 0 .85rem .95rem">'
                            f'<strong style="color:#20233f">{escape(step["title"])}</strong>'
                            f'<p style="color:#626985;font-size:.88rem;margin:.15rem 0 0">{escape(step["detail"])}</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
        else:
            with st.container(border=True):
                st.info(
                    "Your answer and step-by-step updates will appear here after you send a request."
                )

    st.subheader("Recent requests")
    if tickets:
        recent = pd.DataFrame(
            [
                {
                    "Ticket": ticket.get("ticket_id"),
                    "Date": ticket.get("created_at", "")[:16].replace("T", " "),
                    "Question": ticket.get("complaint", ""),
                    "Status": customer_ticket_status(
                        ticket.get("status", "Unknown"),
                        ticket.get("human_review_status"),
                    ),
                }
                for ticket in tickets[:5]
            ]
        )
        st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.caption("Your first support request will appear here.")


def _staff_login() -> tuple[str | None, str | None]:
    staff_key = st.session_state.get("human_review_key")
    reviewer = st.session_state.get("human_review_reviewer", "")
    if staff_key:
        _, error = api_request(
            "GET", "/admin/dashboard", staff_key=staff_key
        )
        if not error:
            return staff_key, reviewer
        st.session_state.pop("human_review_key", None)
        if "HTTP 401" not in error:
            show_api_error(error)

    st.markdown('<div class="eyebrow">STAFF WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Human review sign in")
    st.caption("Use the staff review key configured by the service administrator.")
    with st.form("human_review_sign_in"):
        reviewer = st.text_input("Your name", max_chars=100)
        staff_key = st.text_input("Staff review key", type="password")
        submitted = st.form_submit_button(
            "Sign in to review requests", type="primary", width="stretch"
        )
    if submitted:
        if len(reviewer.strip()) < 2 or not staff_key:
            st.warning("Enter your name and the staff review key.")
        else:
            dashboard, error = api_request(
                "GET", "/admin/dashboard", staff_key=staff_key
            )
            if error:
                show_api_error(error)
            elif isinstance(dashboard, dict):
                st.session_state["human_review_key"] = staff_key
                st.session_state["human_review_reviewer"] = reviewer.strip()
                st.rerun()
    return None, None


def _admin_request(method: str, path: str) -> tuple[Any | None, str | None]:
    return api_request(
        method, path, staff_key=st.session_state["human_review_key"]
    )


def _render_staff_review_cards(reviews: list[dict[str, Any]]) -> None:
    reviewer = st.session_state["human_review_reviewer"]
    if not reviews:
        st.success("There are no requests waiting for human review.")
        return
    st.info(
        "Your decision is recorded for the support team. It does not issue a refund, "
        "change an order, or send an approval message to the customer."
    )
    for item in reviews:
        ticket_id = item.get("ticket_id", "Unknown ticket")
        with st.container(border=True):
            st.markdown(
                f"**{ticket_id}** · {item.get('customer_name') or 'Customer'} · "
                f"{item.get('category', 'support').title()} · "
                f"{item.get('urgency', 'Unknown')} urgency"
            )
            st.caption(item.get("human_review_reason") or item.get("trigger_reason") or "")
            st.markdown("**Customer request**")
            st.write(item.get("complaint") or "No request details are available.")
            if item.get("order_id"):
                st.caption(f"Order: {item['order_id']}")
            with st.expander("Current customer update and support notes"):
                st.write(item.get("response") or "No response recorded.")
                st.caption(f"Ticket status: {item.get('status', 'Unknown')}")
                if item.get("email"):
                    st.caption(f"Customer email: {item['email']}")
            review_status = item.get("status") if "trigger_reason" in item else item.get(
                "human_review_status"
            )
            if review_status and review_status != "Pending":
                st.caption(
                    f"Decision: {review_status} · Reviewer: "
                    f"{item.get('reviewer') or '—'}"
                )
                if item.get("reviewer_note"):
                    st.write(item["reviewer_note"])
                continue
            with st.form(f"human_review_{ticket_id}"):
                note = st.text_area(
                    "Review note (optional)", key=f"review_note_{ticket_id}",
                    max_chars=1000,
                )
                approve_col, reject_col = st.columns(2)
                with approve_col:
                    approve = st.form_submit_button(
                        "Approve review", type="primary", width="stretch"
                    )
                with reject_col:
                    reject = st.form_submit_button(
                        "Reject / needs more work", width="stretch"
                    )
            if approve or reject:
                decision = "Approved" if approve else "Rejected"
                result, error = api_request(
                    "POST",
                    f"/admin/reviews/{ticket_id}",
                    staff_key=st.session_state["human_review_key"],
                    json={
                        "decision": decision,
                        "reviewer": reviewer,
                        "note": note.strip(),
                    },
                )
                if error:
                    show_api_error(error)
                elif isinstance(result, dict):
                    st.success(f"{ticket_id} review recorded: {decision}.")
                    st.rerun()


def render_staff_console() -> None:
    staff_key, _reviewer = _staff_login()
    if not staff_key:
        return

    with st.sidebar:
        st.markdown(f"## {APP_NAME}")
        st.caption("HUMAN-SUPERVISED SUPPORT")
        page = st.radio(
            "Operations",
            ["Dashboard", "Tickets", "Escalations", "Refund reviews", "Audit logs"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Signed in · {st.session_state['human_review_reviewer']}")
        if st.button("Sign out of staff console", width="stretch"):
            st.session_state.pop("human_review_key", None)
            st.session_state.pop("human_review_reviewer", None)
            st.session_state.pop("staff_portal", None)
            st.rerun()

    st.markdown(
        f'<div class="eyebrow">{APP_NAME.upper()} · HUMAN-SUPERVISED OPERATIONS</div>',
        unsafe_allow_html=True,
    )
    st.title(page)
    st.caption("AI-assisted support, grounded in evidence and supervised by staff.")

    if page == "Dashboard":
        dashboard, error = _admin_request("GET", "/admin/dashboard")
        if error:
            show_api_error(error)
            return
        if not isinstance(dashboard, dict):
            st.error("The service returned an invalid dashboard.")
            return
        metric_cols = st.columns(4)
        for col, label, key in zip(
            metric_cols,
            ["Total tickets", "Resolved", "Escalated", "Waiting for review"],
            ["total_tickets", "resolved_tickets", "escalated_tickets", "pending_reviews"],
        ):
            col.metric(label, dashboard.get(key, 0))
        st.subheader("Recent tickets")
        recent = dashboard.get("recent_tickets", [])
        frame = pd.DataFrame([
            {
                "Ticket ID": row.get("ticket_id"),
                "Created": row.get("created_at"),
                "Customer": row.get("customer_name"),
                "Email": row.get("email"),
                "Category": row.get("category"),
                "Urgency": row.get("urgency"),
                "Sentiment": row.get("sentiment"),
                "Status": row.get("human_review_status") or row.get("status"),
                "Escalation reason": row.get("reason"),
            }
            for row in recent
        ])
        st.dataframe(frame, width="stretch", hide_index=True)
        pending, _ = _admin_request("GET", "/admin/reviews?status=Pending")
        if isinstance(pending, list) and pending:
            st.subheader("Needs human review")
            _render_staff_review_cards(pending)
    elif page == "Tickets":
        tickets, error = _admin_request("GET", "/admin/tickets")
        if error:
            show_api_error(error)
            return
        if not isinstance(tickets, list):
            st.error("The service returned an invalid ticket list.")
            return
        st.dataframe(pd.DataFrame([
            {
                "Ticket ID": row.get("ticket_id"),
                "Created": row.get("created_at"),
                "Customer": row.get("customer_name"),
                "Email": row.get("email"),
                "Category": row.get("category"),
                "Urgency": row.get("urgency"),
                "Sentiment": row.get("sentiment"),
                "Status": row.get("status"),
                "Review": row.get("human_review_status") or "—",
                "Escalation reason": row.get("reason"),
            }
            for row in tickets
        ]), width="stretch", hide_index=True)
        st.download_button(
            "Download tickets CSV",
            data=pd.DataFrame(tickets).to_csv(index=False).encode("utf-8-sig"),
            file_name="support_tickets.csv",
            mime="text/csv",
        )
    elif page in {"Escalations", "Refund reviews"}:
        reviews, error = _admin_request("GET", "/admin/reviews")
        if error:
            show_api_error(error)
            return
        if not isinstance(reviews, list):
            st.error("The service returned an invalid review list.")
            return
        if page == "Escalations":
            selected = [item for item in reviews if item.get("status") == "Pending"]
            st.subheader("Escalated requests awaiting a decision")
        else:
            selected = [
                item for item in reviews
                if item.get("category", "").casefold() == "refund"
            ]
            st.subheader("Refund requests requiring staff review")
        _render_staff_review_cards(selected)
    else:
        audit, error = _admin_request("GET", "/admin/audit")
        if error:
            show_api_error(error)
            return
        if not isinstance(audit, list):
            st.error("The service returned invalid audit records.")
            return
        st.dataframe(pd.DataFrame([
            {
                "Time": row.get("created_at"),
                "Ticket ID": row.get("ticket_id"),
                "Action": row.get("action"),
                "Reviewer": row.get("reviewer"),
                "Customer": row.get("customer_name"),
                "Category": row.get("category"),
                "Note": row.get("note"),
            }
            for row in audit
        ]), width="stretch", hide_index=True)


def main() -> None:
    if st.session_state.get("staff_portal"):
        render_staff_console()
        return
    customer = st.session_state.get("customer")
    if not isinstance(customer, dict) or not customer.get("customer_id"):
        st.session_state.pop("customer", None)
        login_page()
        return
    verified_customer, error = api_request(
        "GET", "/me", customer_id=customer["customer_id"]
    )
    if error or not isinstance(verified_customer, dict):
        st.session_state.pop("customer", None)
        st.session_state.pop("latest_ticket", None)
        show_api_error(error or "The selected demo customer is no longer available.")
        login_page()
        return

    customer = verified_customer
    st.session_state["customer"] = customer
    page = render_sidebar(customer)
    if page == "Ask support":
        render_support_page(customer)
    elif page == "My tickets":
        render_ticket_history(customer["customer_id"])
    else:
        st.markdown('<div class="eyebrow">ACCOUNT</div>', unsafe_allow_html=True)
        st.title("Settings")
        with st.container(border=True):
            st.subheader("Demo account")
            st.write(customer.get("name", ""))
            st.caption(customer.get("email", ""))
            st.caption(f"Customer ID · {customer.get('customer_id', '')}")
            st.info(
                "This UI currently connects to the demo API. Real customer sign-in requires a "
                "backend authentication provider and verified session tokens; selecting a demo "
                "account is not production authentication."
            )
        with st.container(border=True):
            st.subheader("Service status")
            health, error = api_request("GET", "/health", customer_id=customer["customer_id"])
            if error:
                show_api_error(error)
            elif health:
                st.write(f"API: {health.get('status', 'unknown')}")
                st.write(f"Mode: {health.get('mode', 'unknown')}")
            st.caption(f"Configured API endpoint: {API_URL}")


if __name__ == "__main__":
    main()
