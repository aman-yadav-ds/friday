from langchain.tools import tool
from utils.tools.music_tool import SpotifyTool
from utils.tools.email_tool import (EmailTool, EmailSessionManager, GmailSender, 
                                    read_email_for_confirmation, confirm_email, send_confirmed_email)
from enum import Enum

class EmailState(Enum):
    DRAFTED = "drafted"
    CONFIRMED = "confirmed"
    SENT = "sent"

@tool
def play_music(query: str) -> str:
    """This tools takes query as the song name by author and plays it on spotify"""
    return SpotifyTool().play_song(query)

@tool
def stop_music() -> str:
    """This tool stops the music playing on spotify"""
    return SpotifyTool().stop_playback()

@tool
def set_volume(level: int) -> str:
    """This tool sets the volume on spotify to the specified level (0-100)"""
    return SpotifyTool().set_volume(level)

@tool
def read_emails(count: int) -> str:
    """This tool reads the latest 'count' emails from the user's inbox"""
    return EmailTool().read_emails(count)

@tool
def reply_email(recipient: str, subject: str, body: str) -> str:
    """This tool sends an email to the specified recipient with the given subject and body"""
    return EmailTool().reply_email(recipient, subject, body)


email_session = EmailSessionManager()
@tool
def generate_email_draft(recipient: str, topic: str) -> str:
    """Generate an email draft and store it for confirmation"""
    draft = EmailTool().generate_email_draft(recipient, topic)
    email_session.set_draft(draft)

    return (
        "I have drafted the email.\n"
        "Say 'read the email' to hear it, "
        "or 'send the email' to confirm."
    )

@tool
def read_draft_email() -> str:
    """Read the drafted email for confirmation"""
    if not email_session.has_draft():
        return "There is no email draft to read."

    return read_email_for_confirmation(email_session.pending_email)

@tool
def confirm_email_send() -> str:
    """Confirm and send the drafted email"""
    if not email_session.has_draft():
        return "There is no email draft to confirm."

    pending = email_session.pending_email
    confirm_email(pending)

    sender = GmailSender()
    send_confirmed_email(pending, sender)

    email_session.clear()

    return "The email has been sent successfully."