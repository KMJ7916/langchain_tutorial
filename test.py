from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

# main.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4-0125-preview"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""

content=""" 평년기온	비슷함
평년강수	비슷함
이동성 고기압의 영향을 받겠으나, 일시적으로 상층 찬 공기의 영향을 받을 때가 있겠습니다. 남쪽을 지나는 기압골의 영향을 받을 때가 있겠습니다. (월평균기온) 평년(5.0~6.0℃)과 비슷하거나 높을 확률이 각각 40% 입니다. (월강수량) 평년(20.0~46.1㎜)과 비슷하거나 많을 확률이 각각 40% 입니다. """

st.header("날씨")
st.info("날씨를 알아볼 수 있는 Q&A 로봇입니다.")
st.error("날씨에 대한 내용이 적용되어 있습니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 날씨 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))