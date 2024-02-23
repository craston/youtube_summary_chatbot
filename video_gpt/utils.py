import requests
from bs4 import BeautifulSoup


def get_video_title(youtube_url: str) -> str:
    r = requests.get(youtube_url)
    soup = BeautifulSoup(r.text, features="html.parser")

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    return title

def video_exists(youtube_url: str) -> bool:
    r = requests.get(youtube_url)
    soup = BeautifulSoup(r.text, features="html.parser")
    if "Video unavailable" in soup.text:
        return False
    return True