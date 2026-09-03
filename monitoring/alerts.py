import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
load_dotenv()

def send_alert(subject, message):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    alert_email = os.getenv("ALERT_EMAIL")

    if not all([
        smtp_server,
        smtp_username,
        smtp_password,
        alert_email
    ]):
        print("Alert configuration not available. Skipping email alert.")
        return

    email = EmailMessage()
    email["Subject"] = subject
    email["From"] = smtp_username
    email["To"] = alert_email
    email.set_content(message)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(email)

    print("Alert email sent successfully.")