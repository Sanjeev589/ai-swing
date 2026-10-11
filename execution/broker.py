"""Explicitly paper-only execution adapter."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: int
    side: str

class PaperBroker:
    def __init__(self):
        self.orders = []

    def place_order(self, order: Order):
        if order.quantity <= 0 or order.side not in {"BUY", "SELL"}:
            raise ValueError("Invalid order")
        self.orders.append(order)
        return {"status": "PAPER_ONLY", "order_number": len(self.orders)}
