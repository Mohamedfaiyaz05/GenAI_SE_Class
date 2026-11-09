"""Run Step 2 repositories without pytest: python run_step2_demo.py"""
from pathlib import Path
from tempfile import TemporaryDirectory
from pprint import pprint

from database.seed import seed_demo_data
from database.order_repository import AuthenticatedCustomer, get_order, list_customer_orders
from database.ticket_repository import NewTicket, save_ticket, list_tickets


def show_case(title: str, inputs: dict, output: object) -> None:
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")
    print('INPUT:')
    pprint(inputs, sort_dicts=False)
    print('OUTPUT:')
    pprint(output, sort_dicts=False)


def main() -> None:
    # A temporary DB keeps this demonstration isolated from real application data.
    with TemporaryDirectory(prefix='shopease_step2_') as folder:
        db_path = Path(folder) / 'demo.db'
        seed_demo_data(db_path)
        print('ShopEase Step 2 demo — isolated SQLite database initialized.')
        jane = AuthenticatedCustomer(customer_id='C001')
        alex = AuthenticatedCustomer(customer_id='C002')

        order = get_order('ORD101', jane, db_path)
        show_case('1. Authorized order lookup', {'customer_id': 'C001', 'order_id': 'ORD101'}, order)
        assert order is not None and order['order_id'] == 'ORD101'

        blocked = get_order('ORD102', jane, db_path)
        show_case('2. Cross-customer lookup is denied', {'customer_id': 'C001', 'order_id': 'ORD102'}, blocked)
        assert blocked is None

        own_order = get_order('ORD102', alex, db_path)
        show_case('3. Actual owner can retrieve order', {'customer_id': 'C002', 'order_id': 'ORD102'}, own_order)
        assert own_order is not None and own_order['order_status'] == 'Delivered'

        orders = list_customer_orders(jane, db_path)
        show_case('4. List customer orders', {'customer_id': 'C001'}, orders)
        assert {item['order_id'] for item in orders} == {'ORD101', 'ORD103'}

        missing = get_order('ORD999', jane, db_path)
        show_case('5. Unknown order', {'customer_id': 'C001', 'order_id': 'ORD999'}, missing)
        assert missing is None

        ticket_input = NewTicket(
            customer_id='C001', customer_name='Jane Doe', order_id='ORD101',
            complaint='My order is delayed!', category='Shipping', urgency='High',
            sentiment='Angry', response='Your concern requires human review.',
            status='Escalated', escalation_reason='High urgency',
            kb_sources=['shipping_policy.md'], evidence_status='PARTIAL'
        )
        ticket_id = save_ticket(ticket_input, db_path)
        saved = list_tickets(db_path)
        show_case('6. Save and read escalated ticket',
                  {'complaint': ticket_input.complaint, 'status': ticket_input.status,
                   'escalation_reason': ticket_input.escalation_reason},
                  {'ticket_id': ticket_id, 'saved_ticket': saved[0]})
        assert len(saved) == 1 and saved[0]['status'] == 'Escalated'

        try:
            save_ticket(NewTicket(
                complaint='Please help', category='Shipping', urgency='High',
                sentiment='Angry', response='Escalating', status='Escalated'
            ), db_path)
        except ValueError as exc:
            show_case('7. Missing escalation reason rejected',
                      {'status': 'Escalated', 'escalation_reason': None},
                      {'error_type': type(exc).__name__, 'message': str(exc)})
        else:
            raise AssertionError('Expected a ValueError for missing escalation reason')

        print('\nALL 7 DEMO CHECKS PASSED. No real application data was modified.')
        print('Note: AuthenticatedCustomer is a trusted-session placeholder, not login authentication.')


if __name__ == '__main__':
    main()
