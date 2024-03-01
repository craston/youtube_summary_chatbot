import os
from dotenv import load_dotenv
load_dotenv()   
assert os.getenv("OPENAI_API_KEY") is not None

from typing import Optional

import gradio as gr

from utils import get_youtube_title, get_youtube_url
from video_query_llm import VideoQueryLLM

bot = VideoQueryLLM()

def initialize_video_query_llm(inp: str) -> tuple[str, str]:
    title = get_youtube_title(inp)
    embed_html = get_youtube_url(inp)

    bot.load_video(inp)
    summary = bot.get_summary()
    
    summary = f"Video Title: {title}\n\nSummary: {summary}"
    return embed_html, summary

def my_chat_function(message: str, history: list[tuple[str, str]]):
    if not hasattr(bot, "retriever"):
        history.append((message, "Please provide a YouTube link first and Process Video"))
        return "", history
    
    response = bot.get_response(message)
    history.append((message, response))
    return "", history

def clear_chat(message, chatbot):
    return "", []

def run():   

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