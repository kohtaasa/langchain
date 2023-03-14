import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('### 読み込んだデータ')
    st.dataframe(df)

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

    with st.container():
        query = st.text_input('このデータについて知りたいことはなんですか？')
        if query:
            ans = agent.run(query)
            st.write(ans)
