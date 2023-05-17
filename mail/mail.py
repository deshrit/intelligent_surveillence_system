import os
import ssl
import smtplib
import imghdr

from loguru import logger
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()

def send_mail(credentials: dict, params: dict) -> None:
    em = EmailMessage()
    em["Subject"] = "!!! INTELLIGENT VIDEO SURVEILLENCE !!!"

    # Plain Text message
    body = """
!!! Weapons Detected - Possible Knife !!!
Location coordinates: 27.694474, 85.318876
"""
    em.set_content(body)

    # HTML message
    body = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Video Surveillence</title>
</head>
<body>
    <h1 style="color:red">!!! Weapons Detected !!!</h1>
    <p>Location coordinates: <i>27.694474, 85.318876</i></p>
    <p>{}</p>
</body>
</html>
    """
    # Update email
    tmp = ""
    for c in params:
        tmp += f"""
<ol>
    <li>{c}:
        <ul>
            <li>Accuracy: {params[c][0]} %</li>
            <li>Numbers: {params[c][-1]}</li>
        </ul>
    </li>
</ol>
"""
    body = body.format(tmp)
    em.add_alternative(body, subtype="html")

    # Attachements
    IMG_DIR = "images"
    imgs = os.listdir(IMG_DIR)
    for img in imgs:
        with open(os.path.join(IMG_DIR, img), "rb") as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name
        em.add_attachment(
            file_data, maintype="image", subtype=file_type, filename=file_name
        )

    # Send Mail
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(
        host=credentials["EMAIL_SERVER"],
        port=credentials["EMAIL_PORT"],
        context=context,
    ) as server:
        server.login(
            user=credentials["EMAIL_SENDER"], password=os.environ.get("EMAIL_AUTH_TOKEN")
        )

        for receiver in credentials["EMAIL_RECEIVERS"]:
            server.sendmail(credentials["EMAIL_SENDER"], receiver, em.as_string())
            logger.info(f"Email sent to {receiver}")
