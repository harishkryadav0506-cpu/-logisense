"""
Email drafting tool.

Generates professional customer service email drafts
based on resolution decisions from the AI agents.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Email templates
TEMPLATES = {
    "refund": """Dear {customer_name},

Thank you for reaching out to us regarding your order {order_id}.

We sincerely apologize for the inconvenience you experienced. After reviewing your case, we have processed a refund for your order.

Resolution Details:
- Order ID: {order_id}
- Resolution: Full Refund Approved
- Details: {resolution}

Your refund will be reflected in your original payment method within 5-7 business days. You will receive a confirmation email once the refund has been processed.

If you have any further questions or concerns, please don't hesitate to reach out to us. We value your business and hope to serve you better in the future.

Best regards,
LogiSense Customer Support Team
Reference: {reference_id}
Date: {date}""",

    "reschedule": """Dear {customer_name},

Thank you for contacting us about your order {order_id}.

We understand the frustration caused by the delivery delay, and we want to assure you that we're taking immediate action to resolve this.

Resolution Details:
- Order ID: {order_id}
- Resolution: Delivery Rescheduled
- Details: {resolution}

We have expedited your delivery and you should receive your package within the next 2-3 business days. You will receive updated tracking information shortly.

As a token of our apology, we have applied a 10% discount coupon to your account for your next purchase.

Thank you for your patience and understanding.

Best regards,
LogiSense Customer Support Team
Reference: {reference_id}
Date: {date}""",

    "escalate": """Dear {customer_name},

Thank you for bringing your concern about order {order_id} to our attention.

We take your feedback very seriously. After careful review, we have escalated your case to our senior resolution team for immediate attention.

Resolution Details:
- Order ID: {order_id}
- Resolution: Escalated to Senior Team
- Details: {resolution}

A dedicated case manager will contact you within 24 hours to personally address your concerns. Your case has been assigned the highest priority.

We sincerely apologize for any inconvenience and assure you that we are committed to resolving this matter to your complete satisfaction.

Best regards,
LogiSense Customer Support Team
Reference: {reference_id}
Date: {date}""",

    "general": """Dear {customer_name},

Thank you for contacting us regarding your order {order_id}.

We have reviewed your case and taken the following action:

Resolution Details:
- Order ID: {order_id}
- Resolution: {resolution}

If you have any additional questions or need further assistance, please don't hesitate to reach out.

Thank you for choosing LogiSense.

Best regards,
LogiSense Customer Support Team
Reference: {reference_id}
Date: {date}""",
}


def draft_email(
    customer_name: str,
    order_id: str,
    resolution: str,
    resolution_type: str = "general",
) -> str:
    """
    Generate a professional customer service email draft.

    This function creates a well-formatted email based on the resolution
    type and details provided by the resolver agent.

    Args:
        customer_name: The customer's full name.
        order_id: The order ID being addressed.
        resolution: Detailed resolution description.
        resolution_type: Type of resolution ('refund', 'reschedule', 
                        'escalate', or 'general'). Determines which
                        email template to use.

    Returns:
        Formatted email text string ready to send.
    """
    # Select template
    template_key = resolution_type.lower() if resolution_type.lower() in TEMPLATES else "general"
    template = TEMPLATES[template_key]

    # Generate reference ID
    reference_id = f"LS-{datetime.now().strftime('%Y%m%d')}-{hash(order_id) % 10000:04d}"

    # Format the email
    email = template.format(
        customer_name=customer_name,
        order_id=order_id,
        resolution=resolution,
        reference_id=reference_id,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    logger.info(
        f"Drafted {template_key} email for {customer_name} "
        f"(order: {order_id}, ref: {reference_id})"
    )

    return email


def draft_email_with_metadata(
    customer_name: str,
    order_id: str,
    resolution: str,
    resolution_type: str = "general",
) -> dict:
    """
    Generate an email draft with metadata.

    Args:
        customer_name: The customer's full name.
        order_id: The order ID being addressed.
        resolution: Detailed resolution description.
        resolution_type: Type of resolution.

    Returns:
        Dict containing:
            - email_body: The formatted email text
            - subject: Suggested email subject line
            - to: Customer name (placeholder for email address)
            - resolution_type: The resolution category used
            - reference_id: Unique reference for tracking
    """
    email_body = draft_email(customer_name, order_id, resolution, resolution_type)

    subject_map = {
        "refund": f"Refund Processed — Order {order_id}",
        "reschedule": f"Delivery Update — Order {order_id}",
        "escalate": f"Priority Case Update — Order {order_id}",
        "general": f"Resolution Update — Order {order_id}",
    }

    template_key = resolution_type.lower() if resolution_type.lower() in subject_map else "general"
    reference_id = f"LS-{datetime.now().strftime('%Y%m%d')}-{hash(order_id) % 10000:04d}"

    return {
        "email_body": email_body,
        "subject": subject_map[template_key],
        "to": customer_name,
        "resolution_type": template_key,
        "reference_id": reference_id,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test email drafting
    print("Testing draft_email (refund):")
    email = draft_email(
        customer_name="Aarav Sharma",
        order_id="ORD-00042",
        resolution="Full refund of $149.99 due to delivery delay exceeding 7 business days.",
        resolution_type="refund",
    )
    print(email)

    print("\n" + "=" * 60)
    print("\nTesting draft_email_with_metadata (escalate):")
    result = draft_email_with_metadata(
        customer_name="Priya Patel",
        order_id="ORD-00099",
        resolution="Case escalated due to high severity complaint and repeated delivery failures.",
        resolution_type="escalate",
    )
    print(f"Subject: {result['subject']}")
    print(f"To: {result['to']}")
    print(f"Reference: {result['reference_id']}")
