import os
from dotenv import load_dotenv
load_dotenv()   
assert os.getenv("OPENAI_API_KEY") is not None

from typing import Optional

import gradio as gr

from utils import video_exists
from utils import get_video_title
from video_query_llm import VideoQueryLLM

def initialize_video_query_llm(youtube_url:str) -> tuple[Optional[str], Optional[str]]:

    # Check if the YouTube link is valid.
    if not video_exists(youtube_url):
        gr.Error("The YouTube video does not exist. Please enter a valid YouTube video URL.")
        return None, None

    global VideoQueryLLM_obj ## Is there a better way to do this?
    VideoQueryLLM_obj =  VideoQueryLLM(youtube_url)

    # Get the URL of the YouTube video.
    url = f"https://www.youtube.com/embed/{youtube_url.split('&')[0].split('=')[1]}"

    # Create the HTML code for the embedded YouTube video.
    embed_html = f"<iframe width='560' height='315' src={url} title='YouTube video player' \
    frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; \
    gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"

    title = get_video_title(youtube_url)

    # Generate summary
    summary = VideoQueryLLM_obj.summary_chain.run(VideoQueryLLM_obj.transcript)
    summary = f"Video Title: {title}\n\nSummary: {summary}"
    return embed_html, summary

def my_chat_function(message: str, history):
    input = {"question": message, }
    if 'VideoQueryLLM_obj' not in globals():
        history.append((message, "Please provide a YouTube link first and Process Video"))
        return "", history
    response = VideoQueryLLM_obj.final_chain.invoke(input)
    VideoQueryLLM_obj.memory.save_context(input, {"answer": response["answer"].content})
    VideoQueryLLM_obj.memory.load_memory_variables({})  
    history.append((message, response["answer"].content))
    return "", history

def clear_chat(message, chatbot):
    return "", []

def run():
    # Is there a way to get rid of global variable?
    if 'VideoQueryLLM_obj' in globals():
        print("VideoQueryLLM_obj already exists... so deleting it.")
        del(VideoQueryLLM_obj)    

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Video GPT-3 Chatbot")
        gr.Markdown("Ask a question about the video and the chatbot will answer it.")
        gr.Markdown("Paste a YouTube link and let the conversation begin!")

        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label = "Enter YouTube URL here.")
                btn = gr.Button(value="Process Video and Generate Summary")
                video = gr.HTML(label=True)
                disp = gr.Textbox(label="Video Summary")
                
            with gr.Column():
                chatbot = gr.Chatbot(label="Ask me anything about the video!")
                msg = gr.Textbox(label="Type your message here.")
                clear = gr.ClearButton([msg, chatbot])

        msg.submit(my_chat_function, [msg, chatbot], [msg, chatbot])
        btn.click(initialize_video_query_llm, inputs=inp, outputs=[video, disp] ).then(clear_chat, [msg, chatbot], [msg, chatbot])

    demo.launch()

if __name__ == "__main__":
    run()