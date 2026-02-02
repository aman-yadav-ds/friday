
from utils.helpers import read_yaml_config
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import json

from pydantic import BaseModel, EmailStr, Field, ValidationError

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


load_dotenv()

from enum import Enum

class EmailState(Enum):
    DRAFTED = "drafted"
    CONFIRMED = "confirmed"
    SENT = "sent"

class EmailDraft(BaseModel):
    recipient: EmailStr
    subject: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=20)

class PendingEmail:
    def __init__(self, draft: EmailDraft):
        self.draft = draft
        self.state = EmailState.DRAFTED

class EmailSessionManager:
    def __init__(self):
        self.pending_email: PendingEmail | None = None

    def has_draft(self) -> bool:
        return self.pending_email is not None

    def set_draft(self, draft: EmailDraft):
        self.pending_email = PendingEmail(draft)

    def clear(self):
        self.pending_email = None


class EmailTool:
    def __init__(self, config_file: str = "config/email_config.yaml"):
        self.config = read_yaml_config(config_file)
        self.email_llm = ChatGroq(
            model=self.config.get("model", "qwen/qwen3-32b"),
            temperature=0.7, 
            max_tokens=1500,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self._router_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You write emails.\n"
                "Return ONLY valid JSON matching this schema:\n"
                "{ recipient: string (email), subject: string, body: string }\n"
                "No markdown. No explanations. No extra keys."
            )),
            ("human", "{input}")
        ])
    
    def read_emails(self, count: int) -> str:
        """
        Read the latest 'count' emails from the user's inbox.
        """
        # Placeholder implementation
        return f"Reading the latest {count} emails from the inbox."

    def reply_email(self, recipient: str, subject: str, body: str) -> str:
        """
        Send an email to the specified recipient with the given subject and body.
        """
        # Placeholder implementation
        return f"Replied to {recipient} with subject '{subject}'."

    def generate_email_draft(self, recipient: str, topic:str) -> str:
        """
        Generate an email body based on the recipient and topic using a predefined template.
        """
        chain = self._router_prompt | self.email_llm
        response = chain.invoke(input=f"User wants to write an email to {recipient} about {topic}.").content    
        
        try:
            email_data = json.loads(response)
        except json.JSONDecodeError:
            return ValueError("LLM returned invalid JSON.")
        

        try:
            return EmailDraft(**email_data)
        except ValidationError as e:
            return RuntimeError(f"Email Draft validation error: {e}")

class GmailSender:
    def __init__(self):
        self.email = os.getenv("GMAIL_ADDRESS")
        self.password = os.getenv("GMAIL_APP_PASSWORD")

        if not self.email or not self.password:
            raise RuntimeError("Missing Gmail credentials")

    def send_email(self, recipient, subject, body):
        msg = MIMEMultipart()
        msg["From"] = self.email
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self.email, self.password)
            server.send_message(msg)



def read_email_for_confirmation(pending: PendingEmail) -> str:
    d = pending.draft
    return (
        f"Recipient: {d.recipient}\n\n"
        f"Subject: {d.subject}\n\n"
        f"Body:\n{d.body}"
    )

def confirm_email(pending: PendingEmail):
    if pending.state != EmailState.DRAFTED:
        raise RuntimeError("Email not in confirmable state")
    pending.state = EmailState.CONFIRMED

def send_confirmed_email(pending: PendingEmail, sender):
    if pending.state != EmailState.CONFIRMED:
        raise RuntimeError("Email not confirmed")

    sender.send_email(
        recipient=pending.draft.recipient,
        subject=pending.draft.subject,
        body=pending.draft.body
    )

    pending.state = EmailState.SENT