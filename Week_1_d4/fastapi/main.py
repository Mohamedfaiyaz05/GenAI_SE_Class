from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# -------------------------
# Request models
# -------------------------

class BusRequest(BaseModel):
    bus_name: str
    source: str
    destination: str
    total_seats: int
    price_per_seat: float


class BookingRequest(BaseModel):
    passenger_name: str
    bus_name: str
    seats: int


# -------------------------
# Response model
# -------------------------

class BookingResponse(BaseModel):
    passenger_name: str
    bus_name: str
    seats_booked: int
    price_per_seat: float
    total_amount: float
    seats_remaining: int
    booking_status: str


# -------------------------
# Temporary database
# -------------------------

buses = {}


# -------------------------
# 1. Add a bus
# -------------------------

@app.post("/buses")
def add_bus(bus: BusRequest):

    buses[bus.bus_name] = {
        "source": bus.source,
        "destination": bus.destination,
        "total_seats": bus.total_seats,
        "available_seats": bus.total_seats,
        "price_per_seat": bus.price_per_seat
    }

    return {
        "message": "Bus added successfully",
        "bus": buses[bus.bus_name]
    }


# -------------------------
# 2. Check available buses
# -------------------------

@app.get("/buses")
def get_buses():
    return buses


# -------------------------
# 3. Book a bus
# -------------------------

@app.post("/book", response_model=BookingResponse)
def book_ticket(booking: BookingRequest):

    # Find bus
    if booking.bus_name not in buses:
        raise HTTPException(
            status_code=404,
            detail="Bus not found"
        )

    bus = buses[booking.bus_name]

    # Check availability
    if booking.seats > bus["available_seats"]:
        raise HTTPException(
            status_code=400,
            detail=f"Only {bus['available_seats']} seats are available"
        )

    # Calculate price
    total_amount = booking.seats * bus["price_per_seat"]

    # Update available seats
    bus["available_seats"] -= booking.seats

    # Return updated information
    return {
        "passenger_name": booking.passenger_name,
        "bus_name": booking.bus_name,
        "seats_booked": booking.seats,
        "price_per_seat": bus["price_per_seat"],
        "total_amount": total_amount,
        "seats_remaining": bus["available_seats"],
        "booking_status": "Confirmed"
    }