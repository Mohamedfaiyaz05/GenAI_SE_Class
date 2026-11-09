from dataclasses import dataclass
from pathlib import Path
from database.connection import connect

@dataclass(frozen=True)
class AuthenticatedCustomer:
    """Only the trusted application/session layer may construct this context."""
    customer_id: str


def get_order(order_id: str, customer: AuthenticatedCustomer, db_path: Path | None = None) -> dict | None:
    if not isinstance(customer, AuthenticatedCustomer) or not customer.customer_id.strip():
        raise ValueError('A trusted authenticated customer context is required')
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError('order_id is required')
    with connect(db_path) as conn:
        row = conn.execute('''SELECT order_id, order_status, item_name,
            estimated_delivery_date, delivered_at, tracking_number,
            tracking_summary, updated_at FROM orders
            WHERE order_id = ? AND customer_id = ?''',
            (order_id.strip(), customer.customer_id)).fetchone()
    return dict(row) if row else None


def list_customer_orders(customer: AuthenticatedCustomer, db_path: Path | None = None) -> list[dict]:
    if not isinstance(customer, AuthenticatedCustomer) or not customer.customer_id.strip():
        raise ValueError('A trusted authenticated customer context is required')
    with connect(db_path) as conn:
        rows = conn.execute('''SELECT order_id, order_status, item_name,
            estimated_delivery_date FROM orders WHERE customer_id = ?
            ORDER BY updated_at DESC, order_id''', (customer.customer_id,)).fetchall()
    return [dict(row) for row in rows]
