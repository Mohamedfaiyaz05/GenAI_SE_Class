"""Shared support-ticket state used by workflow participants."""

from pydantic import BaseModel, Field


class TicketState(BaseModel):
    """Initial typed state for one customer-support request."""

    customer_id: str = ""
    query: str = ""
    intent: str = ""
    order_id: str = ""
    evidence: list[str] = Field(default_factory=list)
    resolution: str = ""
    ticket_id: str = ""
    needs_human: bool = False
