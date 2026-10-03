import streamlit as st
import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bus Booking System",
    page_icon="🚌",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .bus-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }

    .success-box {
        background-color: #ecfdf5;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #10b981;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚌 Bus Booking System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Search buses, check availability and book your ticket</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚌 Bus Booking")

menu = st.sidebar.radio(
    "Choose an option",
    [
        "🏠 Available Buses",
        "➕ Add Bus",
        "🎫 Book Ticket"
    ]
)


# ============================================================
# FUNCTION: GET BUSES FROM FASTAPI
# ============================================================

def get_buses():

    try:

        response = requests.get(
            f"{API_URL}/buses"
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:

        st.error(
            f"Unable to connect to FastAPI: {e}"
        )

        return {}


# ============================================================
# PAGE 1: AVAILABLE BUSES
# ============================================================

if menu == "🏠 Available Buses":

    st.header("Available Buses")

    buses = get_buses()

    if not buses:

        st.info("No buses available.")

    else:

        # buses is a dictionary
        #
        # bus_name  ->  bus_details
        #
        # Example:
        # "Chennai Express" -> {
        #     "source": "Chennai",
        #     "destination": "Cudadlore",
        #     ...
        # }

        for bus_name, bus in buses.items():

            with st.container():

                st.markdown(
                    f"""
                    <div class="bus-card">

                    <h3>🚌 {bus_name}</h3>

                    <p>
                    📍 <b>{bus['source']}</b>
                    →
                    <b>{bus['destination']}</b>
                    </p>

                    <p>
                    💺 Available Seats:
                    <b>{bus['available_seats']}</b>
                    </p>

                    <p>
                    💰 Price:
                    <b>₹{bus['price_per_seat']}</b>
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# PAGE 2: ADD BUS
# ============================================================

elif menu == "➕ Add Bus":

    st.header("Add New Bus")

    with st.form("add_bus_form"):

        bus_name = st.text_input(
            "Bus Name",
            placeholder="Example: Chennai Express"
        )

        col1, col2 = st.columns(2)

        with col1:

            source = st.text_input(
                "Source",
                placeholder="Chennai"
            )

        with col2:

            destination = st.text_input(
                "Destination",
                placeholder="Bangalore"
            )

        col3, col4 = st.columns(2)

        with col3:

            total_seats = st.number_input(
                "Total Seats",
                min_value=1,
                value=40
            )

        with col4:

            price_per_seat = st.number_input(
                "Price Per Seat",
                min_value=1.0,
                value=750.0
            )

        submitted = st.form_submit_button(
            "➕ Add Bus"
        )

    if submitted:

        data = {
            "bus_name": bus_name,
            "source": source,
            "destination": destination,
            "total_seats": total_seats,
            "price_per_seat": price_per_seat
        }

        try:

            response = requests.post(
                f"{API_URL}/buses",
                json=data
            )

            if response.status_code == 200:

                st.success(
                    "✅ Bus added successfully!"
                )

            else:

                st.error(
                    response.json().get(
                        "detail",
                        "Unable to add bus"
                    )
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to FastAPI: {e}"
            )


# ============================================================
# PAGE 3: BOOK TICKET
# ============================================================

elif menu == "🎫 Book Ticket":

    st.header("🎫 Book Your Ticket")

    buses = get_buses()

    if not buses:

        st.warning(
            "No buses available for booking."
        )

    else:

        # Since buses is a dictionary,
        # get the bus names using .keys()

        bus_names = list(buses.keys())

        selected_bus = st.selectbox(
            "Select Bus",
            bus_names
        )

        # Get selected bus details
        selected_bus_data = buses[selected_bus]

        # Display bus information

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Route",
                f"{selected_bus_data['source']} → "
                f"{selected_bus_data['destination']}"
            )

        with col2:

            st.metric(
                "Available Seats",
                selected_bus_data["available_seats"]
            )

        with col3:

            st.metric(
                "Price / Seat",
                f"₹{selected_bus_data['price_per_seat']}"
            )

        st.divider()

        # ====================================================
        # BOOKING FORM
        # ====================================================

        with st.form("booking_form"):

            passenger_name = st.text_input(
                "Passenger Name"
            )

            available_seats = selected_bus_data["available_seats"]

            if available_seats == 0:

                st.warning("⚠️ Seats unavailable — all seats are full.")

            else:

                seats = st.number_input(
                "Number of Seats",
                min_value=1,
                max_value=available_seats,
                value=1
                        )

            book_button = st.form_submit_button(
                "🎫 Book Ticket",disabled=(available_seats == 0)
            )

        # ====================================================
        # SEND BOOKING REQUEST TO FASTAPI
        # ====================================================

        if book_button:

            booking_data = {
                "passenger_name": passenger_name,
                "bus_name": selected_bus,
                "seats": seats
            }

            try:

                response = requests.post(
                    f"{API_URL}/book",
                    json=booking_data
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "🎉 Booking confirmed!"
                    )

                    selected_bus_data["available_seats"] = result["seats_remaining"]
                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Seats Booked",
                            result["seats_booked"]
                        )

                    with col2:

                        st.metric(
                            "Total Amount",
                            f"₹{result['total_amount']}"
                        )

                    with col3:

                        st.metric(
                            "Seats Remaining",
                            result["seats_remaining"]
                        )

                else:

                    error_message = response.json().get(
                        "detail",
                        "Booking failed"
                    )

                    st.error(
                        f"❌ {error_message}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Unable to connect to FastAPI: {e}"
                )
