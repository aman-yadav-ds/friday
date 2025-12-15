from langchain.tools import tool
from utils.tools.music_tool import SpotifyTool
from utils.tools.email_tool import EmailTool

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