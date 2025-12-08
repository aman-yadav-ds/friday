from langchain.tools import tool
from utils.tools.music_tool import SpotifyTool

@tool
def play_music(query: str) -> str:
    """This tools takes query as the song name by author and plays it on spotify"""
    return SpotifyTool().play_song(query)

@tool
def stop_music() -> str:
    """This tool stops the music playing on spotify"""
    return SpotifyTool().stop_playback()
