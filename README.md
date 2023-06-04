# Leveraging Your Own Documents in a Langchain Pipeline
This project highlights how to leverage a ChromaDB vectorstore in a Langchain pipeline to create *drumroll please* a GPT Investment Banker (ergh, I cringed as I wrote that, but alas it's actually pretty practical). You can load in a pdf based document and use it alongside an LLM without the need for fine tuning.

## See it live and in action 📺
[![Tutorial](https://i.imgur.com/M7GcwGH.jpg)](https://youtu.be/u8vQyTzNGVY 'Tutorial')

# Startup 🚀
1. Clone this repo `git clone https://github.com/nicknochnack/LangchainDocuments`
2. Go into the directory `cd LangchainDocuments`
3. Create a virtual environment `python -m venv langchainenv`
4. Activate it:
   - Windows:`.\langchainenv\Scripts\activate`
   - Mac: `source langchainenv/bin/activate`
5. Install the required dependencies `pip install -r requirements.txt`
6. Add your OpenAI APIKey to line 22 of `app.py`
7. Replace `<file path>` with proper path to a pdf.
7. Start the app `streamlit run app.py`  


# Other References 🔗
<p>The main LG Agent used:<a href="https://python.langchain.com/en/latest/modules/agents/toolkits/examples/vectorstore.html">Langchain VectorStore Agents
</a></p>

# Who, When, Why?
👨🏾‍💻 Author: Nick Renotte <br />
📅 Version: 1.?<br />
📜 License: This project is licensed under the MIT License </br>
