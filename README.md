# Vasama 

This is the repository for the [Vasama Telegram Data Analysis Demo](https://vasama.streamlit.app/). Vasama is a customizable data collection and analysis framework. Vasama uses [Ollama](https://ollama.com/) to run local open source LLM models for multimodal data analysis. Applications include political sentiment analysis, OSINT analysis and market analysis. This [Streamlit dashboard](https://vasama.streamlit.app/) is a demonstration of Vasama data analysis results. Data is related to the war in Ukraine and collected from Telegram channels. The data analysis takes into account OSINT information as well as geopolitical sentiments. 

## Basic Components
* Data Collection: Data collection with custom web scrapers and official APIs.
* Data Storage: Storage of media files in S3 and data in MongoDB.
* Data Analysis: Customizable data analysis pipeline. Uses Ollama to run open source LLM models locally. 
* Data Visualization: [Streamlit dashboard](https://vasama.streamlit.app/) of data analysis results.

## Optional Components
* Daily Summary Reports: Daily summaries of data analysis results.
* Data Collection Agent: Tool-using data collection agent. 
* Data Analysis Chatbot: RAG chatbot explaining data analysis results. 
