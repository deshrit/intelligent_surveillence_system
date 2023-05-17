import os
from loguru import logger
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()

# Send message
def send_sms(credentials: dict, params: dict):
    body = """
!!! Weapons Detected !!!
Location coordinates: 27.694474, 85.318876
{}
"""

    tmp = ""
    for c in params:
        tmp += f"""
{c}:
\tAccuracy: {params[c][0]} %
\tNumbers: {params[c][-1]}
"""
    body = body.format(tmp)

    for target in credentials.get("SMS_TARGETS"):
        # Sender Cleint
        client = Client(os.environ.get("SMS_ACCOUNT_SID"), os.environ.get("SMS_AUTH_TOKEN"))
        client.messages.create(from_=credentials["SMS_SENDER"], to=target, body=body)
        logger.info(f"SMS sent to target: {target}")
