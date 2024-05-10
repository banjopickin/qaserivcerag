I created a QA service using RAG technique
![plot](./rag.png)

## Run the app
Follow the steps below to run the app:

1. Navigate to `code` folder in your terminal 
2. Create .env file under `code` folder and put your OPENAI_API_KEY in it 
3. Make sure you have Docker installed on your machine, if not please download one
4. Run `docker build -t e104_final .` to build the container
5. Run `docker run --env-file ./.env -p 8501:8501 e104_final` to run the container. It will show a URL in the terminal 
6. Copy and paste the URL to your browser, you’ll see the app. It may take one minute to load the page, but won’t be too long
