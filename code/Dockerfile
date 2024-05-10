FROM python:3.10-slim-bullseye

RUN pip install bs4
RUN pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai

RUN pip install pysqlite3-binary
RUN pip install --upgrade --quiet langchain-chroma
RUN pip install streamlit

WORKDIR /app
COPY src/ /app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]