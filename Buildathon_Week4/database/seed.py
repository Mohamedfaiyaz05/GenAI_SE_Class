"""Starter module for loading local demonstration data."""

from database.connection import connect, initialize_database

CUSTOMERS = [
    ('C001', 'Jane Doe', 'jane@example.test'),
    ('C002', 'Alex Smith', 'alex@example.test'),
    ('C003', 'John Carter', 'john@example.test'),
    ('C004', 'Emily Wilson', 'emily@example.test'),
    ('C005', 'Michael Brown', 'michael@example.test'),
    ('C006', 'Sarah Johnson', 'sarah@example.test'),
    ('C007', 'David Miller', 'david@example.test'),
    ('C008', 'Sophia Davis', 'sophia@example.test'),
    ('C009', 'Daniel Anderson', 'daniel@example.test'),
    ('C010', 'Olivia Taylor', 'olivia@example.test'),
    ('C011', 'James Thomas', 'james@example.test'),
    ('C012', 'Emma Martinez', 'emma@example.test'),
    ('C013', 'William Garcia', 'william@example.test'),
    ('C014', 'Ava Robinson', 'ava@example.test'),
    ('C015', 'Noah Clark', 'noah@example.test'),
]

ORDERS = [
    # Jane Doe
    ('ORD101', 'C001', 'In Transit', 'Wireless Headphones',
     '2026-11-08', None, 'TRK101', 'Regional sorting hub'),

    ('ORD103', 'C001', 'Processing', 'Coffee Maker',
     '2026-11-15', None, None, 'Preparing for dispatch'),

    ('ORD104', 'C001', 'Delivered', 'Bluetooth Speaker',
     '2026-11-02', '2026-11-02', 'TRK104', 'Delivered successfully'),

    # Alex Smith
    ('ORD102', 'C002', 'Delivered', 'Running Shoes',
     '2026-11-05', '2026-11-05', 'TRK102', 'Delivered'),

    ('ORD105', 'C002', 'Cancelled', 'Smart Watch',
     '2026-11-12', None, None, 'Cancelled by customer'),

    # John Carter
    ('ORD106', 'C003', 'In Transit', 'Gaming Mouse',
     '2026-11-11', None, 'TRK106', 'Arrived at distribution center'),

    ('ORD107', 'C003', 'Delivered', 'Mechanical Keyboard',
     '2026-11-04', '2026-11-04', 'TRK107', 'Delivered to customer'),

    # Emily Wilson
    ('ORD108', 'C004', 'Out for Delivery', 'Laptop Backpack',
     '2026-11-09', None, 'TRK108', 'Out for delivery today'),

    ('ORD109', 'C004', 'Returned', 'Wireless Charger',
     '2026-11-01', '2026-11-01', 'TRK109', 'Returned due to defective product'),

    # Michael Brown
    ('ORD110', 'C005', 'Processing', 'USB-C Hub',
     '2026-11-14', None, None, 'Order confirmed'),

    ('ORD111', 'C005', 'Delivered', 'External SSD',
     '2026-11-03', '2026-11-03', 'TRK111', 'Delivered successfully'),

    # Sarah Johnson
    ('ORD112', 'C006', 'In Transit', 'Air Fryer',
     '2026-11-12', None, 'TRK112', 'Package departed warehouse'),

    ('ORD113', 'C006', 'Refund Initiated', 'Hair Dryer',
     '2026-11-02', '2026-11-02', 'TRK113', 'Refund request under processing'),

    # David Miller
    ('ORD114', 'C007', 'Delivered', 'Fitness Tracker',
     '2026-11-06', '2026-11-06', 'TRK114', 'Delivered'),

    ('ORD115', 'C007', 'Cancelled', 'Electric Kettle',
     '2026-11-13', None, None, 'Order cancelled before shipment'),

    # Sophia Davis
    ('ORD116', 'C008', 'Out for Delivery', 'Yoga Mat',
     '2026-11-09', None, 'TRK116', 'Courier assigned for delivery'),

    ('ORD117', 'C008', 'Processing', 'Water Bottle',
     '2026-11-16', None, None, 'Packing in progress'),

    # Daniel Anderson
    ('ORD118', 'C009', 'In Transit', 'Noise Cancelling Earbuds',
     '2026-11-10', None, 'TRK118', 'Shipment reached local hub'),

    ('ORD119', 'C009', 'Delivered', 'Phone Case',
     '2026-11-01', '2026-11-01', 'TRK119', 'Delivered'),

    # Olivia Taylor
    ('ORD120', 'C010', 'Returned', 'Portable Monitor',
     '2026-11-03', '2026-11-03', 'TRK120', 'Return received at warehouse'),

    ('ORD121', 'C010', 'Processing', 'Desk Lamp',
     '2026-11-15', None, None, 'Awaiting dispatch'),

    # James Thomas
    ('ORD122', 'C011', 'Delivered', 'Office Chair',
     '2026-11-05', '2026-11-05', 'TRK122', 'Delivered successfully'),

    ('ORD123', 'C011', 'In Transit', 'Standing Desk',
     '2026-11-14', None, 'TRK123', 'Shipment moving to destination'),

    # Emma Martinez
    ('ORD124', 'C012', 'Refund Initiated', 'Tablet',
     '2026-11-04', '2026-11-04', 'TRK124', 'Refund initiated after return'),

    ('ORD125', 'C012', 'Out for Delivery', 'Tablet Cover',
     '2026-11-09', None, 'TRK125', 'Out for delivery'),

    # William Garcia
    ('ORD126', 'C013', 'Processing', 'Webcam',
     '2026-11-17', None, None, 'Order placed successfully'),

    ('ORD127', 'C013', 'Delivered', 'USB Microphone',
     '2026-11-06', '2026-11-06', 'TRK127', 'Delivered'),

    # Ava Robinson
    ('ORD128', 'C014', 'In Transit', 'Smart Home Camera',
     '2026-11-13', None, 'TRK128', 'Package at regional warehouse'),

    ('ORD129', 'C014', 'Cancelled', 'LED Strip Lights',
     '2026-11-12', None, None, 'Cancelled before dispatch'),

    # Noah Clark
    ('ORD130', 'C015', 'Delivered', 'Power Bank',
     '2026-11-07', '2026-11-07', 'TRK130', 'Delivered successfully'),
]


def seed_demo_data(db_path=None) -> None:
    initialize_database(db_path)

    with connect(db_path) as conn:

        conn.executemany(
            '''
            INSERT OR IGNORE INTO customers
            (customer_id, display_name, email)
            VALUES (?, ?, ?)
            ''',
            CUSTOMERS
        )

        conn.executemany(
            '''
            INSERT OR IGNORE INTO orders
            (
                order_id,
                customer_id,
                order_status,
                item_name,
                estimated_delivery_date,
                delivered_at,
                tracking_number,
                tracking_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            ORDERS
        )


if __name__ == '__main__':
    seed_demo_data()
    print('Demo database initialized and seeded.')
