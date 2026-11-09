-- Starter database schema; add tables and constraints during implementation.
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    order_status TEXT NOT NULL CHECK (order_status IN ('Processing','Shipped','In Transit','Delivered','Cancelled','Returned')),
    item_name TEXT NOT NULL,
    estimated_delivery_date TEXT,
    delivered_at TEXT,
    tracking_number TEXT,
    tracking_summary TEXT,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id),
    customer_name TEXT,
    order_id TEXT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    complaint TEXT NOT NULL,
    category TEXT NOT NULL,
    urgency TEXT NOT NULL CHECK (urgency IN ('Low','Medium','High')),
    sentiment TEXT NOT NULL,
    kb_sources TEXT NOT NULL DEFAULT '[]',
    response TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('Resolved','Escalated')),
    escalation_reason TEXT,
    evidence_status TEXT CHECK (evidence_status IN ('FULL','PARTIAL','NONE','CONFLICTING') OR evidence_status IS NULL),
    answer_attempts INTEGER NOT NULL DEFAULT 0 CHECK (answer_attempts >= 0),
    CHECK (status != 'Escalated' OR (escalation_reason IS NOT NULL AND length(trim(escalation_reason)) > 0))
);
CREATE INDEX IF NOT EXISTS idx_tickets_status_timestamp ON tickets(status, timestamp DESC);
