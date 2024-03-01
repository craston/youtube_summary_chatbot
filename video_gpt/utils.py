import requests
from bs4 import BeautifulSoup

def get_youtube_url(url: str) -> str:
    # Get the URL of the YouTube video.
    url = f"https://www.youtube.com/embed/{url.split('&')[0].split('=')[1]}"

    # Create the HTML code for the embedded YouTube video.
    embed_html = f"<iframe width='560' height='315' src={url} title='YouTube video player' \
    frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; \
    gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"

    return embed_html

def get_youtube_title(youtube_url: str) -> str:
    r = requests.get(youtube_url)
    soup = BeautifulSoup(r.text, features="html.parser")
    if "Video unavailable" in soup.text:
        RuntimeError("Video is unavailable")

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    return title