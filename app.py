import streamlit as st
import pandas as pd
import re
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import sys

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください")


class Logger():
    stdout = sys.stdout
    messages = []

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.stdout

    def write(self, text):
        self.messages.append(text)


log = Logger()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('### 読み込んだデータ')
    st.dataframe(df)

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

    with st.container():
        query = st.text_input('このデータについて知りたいことはなんですか？')
        if query:
            log.start()
            ans = agent.run(query)
            log.stop()
            value = list(log.messages)
            st.write(f'回答: {ans}')
            st.write('')
            if len(value) > 0:
                match = re.search(r'Action Input: (.+?)\x1b', value[2])
                if match:
                    value = match.group(1)
                    st.write(f'実行されたコード: {value}')

