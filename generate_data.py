"""
Synthetic Data Generator for LogiSense
Generates orders.csv, reviews.csv, and policy PDFs.
Run: python generate_data.py
"""

import csv
import random
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Ananya", "Diya", "Priya", "Kavya", "Isha", "Meera",
    "Riya", "Saanvi", "Aanya", "Navya", "James", "Emma", "Liam", "Olivia",
    "Noah", "Ava", "William", "Sophia", "Benjamin", "Isabella", "Lucas", "Mia",
    "Henry", "Charlotte", "Alexander", "Amelia", "Daniel", "Harper", "Matthew",
    "Evelyn", "Carlos", "Maria", "Yuki", "Haruto", "Chen", "Wei", "Fatima",
    "Ahmed", "Priyanka", "Rahul"
]

LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Reddy", "Gupta", "Joshi", "Verma",
    "Yadav", "Mehta", "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "Nakamura", "Wang", "Li", "Hassan", "Ali", "Kim", "Park", "Santos",
    "Fernandez", "Nguyen"
]

PRODUCTS = [
    "Wireless Bluetooth Headphones", "Laptop Stand Adjustable", "USB-C Hub 7-in-1",
    "Mechanical Keyboard RGB", "Portable SSD 1TB", "Webcam HD 1080p",
    "Smart Watch Fitness Tracker", "Noise Cancelling Earbuds", "Tablet Stand Holder",
    "External Battery 20000mAh", "Monitor 27-inch 4K", "Wireless Mouse Ergonomic",
    "Phone Case Premium Leather", "Desk Lamp LED Dimmable", "Cable Management Kit",
    "Microphone USB Condenser", "Ring Light 12-inch", "Graphics Tablet Drawing",
    "Router WiFi 6 Mesh", "Surge Protector 8-Outlet", "Smart Speaker Voice Assistant",
    "Action Camera 4K Waterproof", "Drone Mini Foldable", "VR Headset Standalone",
    "E-Reader 8-inch Display", "Portable Projector Mini", "Smart Home Hub",
    "Electric Toothbrush Sonic", "Air Purifier HEPA Filter", "Robot Vacuum Cleaner"
]

CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "BlueDart", "Delhivery", "Amazon Logistics"]

STATUSES = ["delivered", "in_transit", "delayed", "returned", "cancelled", "out_for_delivery"]

DELAY_REASONS = [
    "weather_disruption", "customs_clearance", "warehouse_backlog",
    "carrier_issue", "address_error", "out_of_stock",
    "vehicle_breakdown", "high_demand_surge", "incorrect_routing",
    "security_check", None, None, None  # None = no delay
]

COMPLAINT_TEMPLATES_LOW = [
    "My order {oid} arrived a day late but everything looks fine. Just wanted to let you know.",
    "The packaging for order {oid} was slightly damaged but the product inside is okay.",
    "Order {oid} tracking was not updating for a day. It arrived though.",
    "I received order {oid} and the color is slightly different from the picture. Not a big deal.",
    "Delivery for {oid} was left at the doorstep without notification. Please improve this.",
    "Order {oid} was supposed to arrive yesterday but came today. Minor inconvenience.",
    "The invoice for order {oid} has a small typo in my name. Can you correct it?",
    "Order {oid} arrived fine but the delivery person was a bit rude.",
    "I got order {oid} a few hours after the estimated window. Not urgent but noting it.",
    "The product from order {oid} works but the manual was missing. Can you send one?",
    "Order {oid} was delivered to the right address but wrong apartment number initially.",
    "Tracking for order {oid} showed delivered but I found it at neighbor's place.",
    "Product from {oid} is good but took 1 extra day compared to the estimate.",
    "Order {oid}: the box was a bit crushed but product is fine, thanks.",
    "Minor issue with order {oid} - the accessories were in a separate bag, initially thought missing.",
]

COMPLAINT_TEMPLATES_MEDIUM = [
    "Order {oid} is delayed by 3 days now. I need it for an event this weekend. Please expedite.",
    "I received a wrong item in order {oid}. I ordered headphones but got a mouse. Need exchange.",
    "Order {oid} has been stuck in transit for 5 days. No tracking updates. Very frustrating.",
    "The product from order {oid} stopped working after 2 days. I want a replacement.",
    "Order {oid} was marked delivered but I haven't received it. Delivery agent marked wrong.",
    "I've been waiting for order {oid} refund for a week now. When will it be processed?",
    "Order {oid} arrived with missing parts. The charger was not included in the box.",
    "The quality of the product in order {oid} doesn't match the description. Want a return.",
    "Order {oid} delivery was rescheduled twice without my consent. Very inconvenient.",
    "I paid for express shipping on order {oid} but it's being shipped standard. Need correction.",
    "Order {oid} product has a scratch on the screen. It's supposed to be brand new.",
    "My order {oid} was split into two shipments without notice. One part is still missing.",
    "Order {oid} was delivered but the seal was broken. I suspect it's a returned product.",
    "I need to cancel order {oid} but the app won't let me. It hasn't shipped yet.",
    "Order {oid}: product arrived but doesn't match the size specifications listed.",
]

COMPLAINT_TEMPLATES_HIGH = [
    "URGENT: Order {oid} never arrived and I was charged twice! I need immediate refund for both charges.",
    "Order {oid} is 10 days late and nobody from support is helping. This is unacceptable! Escalate NOW.",
    "I received a completely damaged product in order {oid}. Screen is cracked. DEMAND full refund.",
    "Order {oid} - I've called 5 times about my missing package. Each time told to wait. I want my money back NOW.",
    "FRAUD ALERT: Order {oid} was delivered to wrong address and signed by unknown person. Need investigation.",
    "Order {oid} product is DEFECTIVE and almost caused a safety hazard. This needs immediate attention.",
    "I've been given the runaround for 2 weeks about order {oid} refund. Filing complaint with consumer forum.",
    "Order {oid} is a SCAM - the product is counterfeit, not genuine. I want full refund and compensation.",
    "EXTREMELY DISAPPOINTED with order {oid}. Wrong product, damaged box, late delivery. Want to speak to manager.",
    "Order {oid} has been lost by the carrier and nobody takes responsibility. 15 days and counting!",
    "Order {oid}: Received used item sold as new. Battery is degraded. This is FRAUD. Refund immediately.",
    "My order {oid} worth $500+ has vanished. No tracking, no updates, no support. Considering legal action.",
    "Order {oid} delivered broken product. Customer service hung up on me TWICE. Absolutely furious.",
    "URGENT: Order {oid} was supposed to be a gift. It's 2 weeks late and the occasion has passed. Full refund.",
    "Order {oid}: Allergic reaction from product that had wrong ingredients listed. Need compensation NOW.",
]

RESOLUTION_TYPES = {
    "low": ["no_action", "apology_email", "coupon_issued"],
    "medium": ["replacement_sent", "partial_refund", "rescheduled_delivery", "exchange_initiated"],
    "high": ["full_refund", "escalated_to_manager", "refund_with_compensation", "investigation_opened"],
}


def random_date(start_year: int = 2024, end_year: int = 2025) -> datetime:
    """Generate a random date within the given year range."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_orders(num_rows: int = 1200) -> list[dict]:
    """Generate synthetic order data."""
    orders = []
    for i in range(1, num_rows + 1):
        order_id = f"ORD-{i:05d}"
        customer = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        product = random.choice(PRODUCTS)
        order_date = random_date()
        status = random.choice(STATUSES)
        carrier = random.choice(CARRIERS)

        # Delivery date logic
        if status == "delivered":
            delivery_date = order_date + timedelta(days=random.randint(1, 14))
        elif status in ("returned", "cancelled"):
            delivery_date = ""
        else:
            delivery_date = ""

        # Delay reason logic
        if status in ("delayed", "returned"):
            delay_reason = random.choice([r for r in DELAY_REASONS if r is not None])
        elif status == "delivered" and random.random() < 0.2:
            delay_reason = random.choice([r for r in DELAY_REASONS if r is not None])
        else:
            delay_reason = ""

        orders.append({
            "order_id": order_id,
            "customer_name": customer,
            "product": product,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "status": status,
            "carrier": carrier,
            "delivery_date": delivery_date.strftime("%Y-%m-%d") if delivery_date else "",
            "delay_reason": delay_reason,
        })
    return orders


def generate_reviews(orders: list[dict], num_rows: int = 600) -> list[dict]:
    """Generate synthetic complaint/review data linked to orders."""
    reviews = []
    # Pick a subset of orders that have complaints
    complaint_orders = random.sample(orders, min(num_rows, len(orders)))

    for i, order in enumerate(complaint_orders, 1):
        severity = random.choices(
            ["low", "medium", "high"],
            weights=[0.3, 0.45, 0.25],
            k=1
        )[0]

        templates = {
            "low": COMPLAINT_TEMPLATES_LOW,
            "medium": COMPLAINT_TEMPLATES_MEDIUM,
            "high": COMPLAINT_TEMPLATES_HIGH,
        }

        complaint_text = random.choice(templates[severity]).format(oid=order["order_id"])
        resolution_type = random.choice(RESOLUTION_TYPES[severity])

        reviews.append({
            "complaint_id": f"CMP-{i:05d}",
            "order_id": order["order_id"],
            "complaint_text": complaint_text,
            "severity": severity,
            "resolution_type": resolution_type,
        })

    return reviews


def _build_pdf(title: str, subtitle: str, sections: list, output_path: str) -> None:
    """Helper to build a formatted policy PDF with proper margins."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, subtitle, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    for section_title, items in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(effective_width, 10, section_title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        for item in items:
            bullet_text = f"  - {item}"
            pdf.multi_cell(effective_width, 6, bullet_text)
            pdf.ln(1)
        pdf.ln(4)

    pdf.output(output_path)


def generate_policy_pdfs() -> None:
    """Generate 3 policy PDF documents using fpdf2."""
    try:
        from fpdf import FPDF  # noqa: F401
    except ImportError:
        print("fpdf2 not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "fpdf2"])

    os.makedirs(POLICIES_DIR, exist_ok=True)

    # ── Refund Policy ──
    refund_sections = [
        ("1. Refund Eligibility", [
            "Customers are eligible for a full refund if the order is cancelled before shipment.",
            "Orders delayed by more than 7 business days are automatically eligible for a full refund.",
            "Orders delayed by 3-7 business days are eligible for a partial refund of 25% of the order value.",
            "Defective or damaged products are eligible for a full refund within 30 days of delivery.",
            "Wrong items delivered qualify for immediate full refund plus return shipping coverage.",
            "Digital products and gift cards are non-refundable once activated or redeemed.",
        ]),
        ("2. Refund Timeframes", [
            "Refund requests must be submitted within 30 days of the delivery date.",
            "For delayed orders, refund eligibility begins from the original estimated delivery date.",
            "Refunds are processed within 5-7 business days after approval.",
            "Credit card refunds may take an additional 3-5 business days to reflect in the statement.",
            "UPI and digital wallet refunds are processed within 24-48 hours.",
            "Bank transfer refunds require 7-10 business days for processing.",
        ]),
        ("3. Partial Refund Conditions", [
            "Orders with minor packaging damage but intact products receive a 10% courtesy refund.",
            "Late deliveries of 1-2 days receive a 15% refund on shipping charges.",
            "If only part of a multi-item order is affected, refund applies to the affected items only.",
            "Opened software or sealed products may receive up to 50% refund at management discretion.",
            "Subscription services are refunded on a pro-rata basis for unused period.",
        ]),
        ("4. Non-Refundable Items", [
            "Perishable goods and food items are non-refundable.",
            "Customized or personalized products cannot be refunded unless defective.",
            "Items marked as Final Sale or Clearance are non-refundable.",
            "Products with removed tags, missing original packaging, or signs of use.",
            "Hazardous materials and flammable products due to safety regulations.",
        ]),
        ("5. Refund Process", [
            "Step 1: Submit refund request through the app or contact customer support.",
            "Step 2: Provide order ID, reason for refund, and supporting evidence if applicable.",
            "Step 3: Our team reviews the request within 24 hours.",
            "Step 4: If approved, refund is initiated to the original payment method.",
            "Step 5: Confirmation email is sent with refund transaction details.",
            "For orders above $200, a supervisor review is required before approval.",
        ]),
        ("6. Dispute Resolution", [
            "If a refund request is denied, customers can file an appeal within 15 days.",
            "Appeals are reviewed by a senior resolution team within 48 hours.",
            "Customers may request a callback from a supervisor for disputed refunds.",
            "All dispute communications are documented for quality assurance.",
            "Final escalation can be made to the Consumer Grievance Portal.",
        ]),
    ]
    _build_pdf(
        "LogiSense Refund Policy",
        "Effective Date: January 1, 2024 | Version 3.2",
        refund_sections,
        os.path.join(POLICIES_DIR, "refund_policy.pdf"),
    )
    print("  Created: refund_policy.pdf")

    # ── Shipping SLA ──
    shipping_sections = [
        ("1. Standard Shipping", [
            "Standard shipping delivers within 5-7 business days from the date of dispatch.",
            "Orders placed before 2:00 PM are dispatched the same business day.",
            "Weekend and holiday orders are dispatched on the next business day.",
            "Standard shipping is available for all domestic orders.",
            "Tracking number is provided within 24 hours of dispatch via email and SMS.",
            "Estimated delivery dates are calculated excluding weekends and public holidays.",
        ]),
        ("2. Express Shipping", [
            "Express shipping guarantees delivery within 2-3 business days.",
            "Express orders placed before 12:00 PM receive same-day dispatch.",
            "Express shipping includes real-time tracking and delivery confirmation.",
            "If express delivery exceeds 3 business days, shipping charges are fully refunded.",
            "Express shipping is available for orders under 30kg and within serviceable PIN codes.",
            "Priority handling ensures express packages are processed first at distribution centers.",
        ]),
        ("3. Same-Day Delivery", [
            "Same-day delivery is available in select metro cities.",
            "Orders must be placed before 10:00 AM for same-day delivery eligibility.",
            "Same-day delivery window is between 6:00 PM and 10:00 PM on the order date.",
            "If same-day delivery fails, the order is upgraded to express with full refund of premium.",
            "Same-day delivery is not available for items shipped from outside the metro area.",
        ]),
        ("4. Carrier SLA Commitments", [
            "FedEx: 99.5% on-time delivery rate, maximum 2-day delay tolerance.",
            "UPS: 99.2% on-time delivery rate, maximum 3-day delay tolerance.",
            "DHL: 98.8% on-time delivery rate, maximum 3-day delay tolerance for international.",
            "BlueDart: 98.5% on-time delivery rate for domestic deliveries.",
            "Delhivery: 97.5% on-time delivery rate, specialized in Tier-2 and Tier-3 cities.",
            "Amazon Logistics: 99.0% on-time delivery rate for Prime-eligible orders.",
            "USPS: 97.0% on-time delivery rate, economical option for lightweight packages.",
        ]),
        ("5. Delay Compensation", [
            "Delays of 1-2 business days: Customer receives a 10% discount coupon on next order.",
            "Delays of 3-5 business days: Full shipping charges refunded automatically.",
            "Delays of 5-7 business days: 25% refund on order value plus shipping refund.",
            "Delays exceeding 7 business days: Full refund eligibility or free express re-shipment.",
            "Weather-related delays: Compensation starts after 3 additional days beyond weather event.",
            "Customs-related delays: Compensation starts after 5 additional days.",
            "Carrier-attributable delays are penalized per the carrier SLA agreement.",
        ]),
        ("6. Delivery Attempt Policy", [
            "A maximum of 3 delivery attempts are made for each package.",
            "If the customer is unavailable, a delivery notification is left.",
            "After 3 failed attempts, the package is held at the nearest facility for 7 days.",
            "Customer can reschedule delivery within the 7-day hold period at no additional cost.",
            "After the hold period, the package is returned to the warehouse and a refund is initiated.",
        ]),
        ("7. International Shipping", [
            "International orders are delivered within 10-15 business days.",
            "Customs duties and import taxes are the responsibility of the customer.",
            "International tracking is available through the carriers global network.",
            "Restricted items list varies by destination country.",
            "International returns follow the destination countrys return policy timeline.",
        ]),
    ]
    _build_pdf(
        "LogiSense Shipping SLA",
        "Effective Date: January 1, 2024 | Version 2.8",
        shipping_sections,
        os.path.join(POLICIES_DIR, "shipping_sla.pdf"),
    )
    print("  Created: shipping_sla.pdf")

    # ── Return Policy ──
    return_sections = [
        ("1. Return Window", [
            "Products can be returned within 30 days of delivery for a full refund.",
            "Electronics and appliances have an extended 45-day return window.",
            "Fashion and apparel items can be returned within 15 days.",
            "The return window begins from the date the package is marked as delivered.",
            "Holiday purchases (Nov 15 - Dec 31) have an extended return window until January 31.",
            "Subscription boxes have a 7-day return window from delivery.",
        ]),
        ("2. Return Condition Requirements", [
            "Items must be in original, unused condition with all tags attached.",
            "Original packaging must be intact and included with the return.",
            "All accessories, manuals, and free gifts must be included.",
            "Products showing signs of use, wear, or damage are not eligible for return.",
            "Electronics must include all cables, chargers, and original documentation.",
            "Hygiene-sensitive products must have sealed packaging intact.",
        ]),
        ("3. Return Process", [
            "Step 1: Initiate return through the LogiSense app or website.",
            "Step 2: Select reason for return and upload photos if product is damaged.",
            "Step 3: Schedule a pickup or drop off at nearest collection point.",
            "Step 4: Pickup is arranged within 2-3 business days of return initiation.",
            "Step 5: Refund or exchange is processed within 5-7 business days of receiving the item.",
            "Return shipping is free for defective products and wrong deliveries.",
            "For other returns, a flat return shipping fee of $5 is deducted from refund.",
        ]),
        ("4. Exchange Policy", [
            "Exchanges are available for size, color, or variant changes within the return window.",
            "Exchange requests are processed within 3-5 business days.",
            "If the requested exchange item is out of stock, a full refund is issued.",
            "Price differences in exchanges are charged or refunded accordingly.",
            "Each order is eligible for a maximum of 2 exchange requests.",
            "Exchanged items carry a new 30-day return window from delivery.",
        ]),
        ("5. Non-Returnable Items", [
            "Gift cards and store credits are non-returnable.",
            "Downloaded digital content and software licenses cannot be returned.",
            "Perishable items, flowers, and food products are non-returnable.",
            "Items purchased during flash sales marked No Return are final.",
            "Customized products with personalization are non-returnable unless defective.",
            "Products with tampered serial numbers or warranty seals are non-returnable.",
        ]),
        ("6. Damaged in Transit", [
            "If a product arrives damaged, report within 48 hours of delivery.",
            "Include photos showing the damage to packaging and product.",
            "Damaged in transit claims are resolved within 24-48 hours.",
            "Customer receives either a replacement or full refund at their choice.",
            "No return shipping required for damaged items.",
            "Insurance claims for high-value damaged items are handled by our logistics team.",
        ]),
        ("7. Quality Guarantee", [
            "All products carry a LogiSense Quality Guarantee for authentic merchandise.",
            "If a product is found to be counterfeit, full refund plus 20% compensation is provided.",
            "Quality complaints are escalated to the brand or seller within 24 hours.",
            "Repeated quality issues from a seller result in seller suspension from the platform.",
        ]),
    ]
    _build_pdf(
        "LogiSense Return Policy",
        "Effective Date: January 1, 2024 | Version 2.5",
        return_sections,
        os.path.join(POLICIES_DIR, "return_policy.pdf"),
    )
    print("  Created: return_policy.pdf")


def main() -> None:
    """Main entry point for data generation."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(POLICIES_DIR, exist_ok=True)

    random.seed(42)  # Reproducibility

    # Generate orders
    print("Generating orders.csv...")
    orders = generate_orders(1200)
    orders_path = os.path.join(DATA_DIR, "orders.csv")
    with open(orders_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=orders[0].keys())
        writer.writeheader()
        writer.writerows(orders)
    print(f"  Created: orders.csv ({len(orders)} rows)")

    # Generate reviews
    print("Generating reviews.csv...")
    reviews = generate_reviews(orders, 600)
    reviews_path = os.path.join(DATA_DIR, "reviews.csv")
    with open(reviews_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)
    print(f"  Created: reviews.csv ({len(reviews)} rows)")

    # Generate PDFs
    print("Generating policy PDFs...")
    generate_policy_pdfs()

    print("\nAll synthetic data generated successfully!")


if __name__ == "__main__":
    main()
