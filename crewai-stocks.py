import json
import os

from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool

from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

#Criando Tool Yahoo Finance Tool
def fetch_stock_price(ticket):
    stock = yf.download(
        ticket, 
        start = "2023-01-01", 
        end = datetime.today()
    )
    
    return stock

yahoo_finace_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks princes for {ticket} from the last year about specific from Yahoo Finance API",
    func = lambda ticket: fetch_stock_price(ticket)
)

#Importando OpenAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model = "gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role =  "Senior stock price Ananlyst",
    goal = "Find the {ticket} stock prince and analyses trends",
    backstory = """You're a highly experienced in analyzing the price of an specific stock
    and make predictions about its future price.""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = False,
    tools = [yahoo_finace_tool]
)

getStockPrice = Task(
    agent = stockPriceAnalyst,
    description = "Analyze the stock {ticket} price history and create a trend analyses of up, donw or sideways.",
    expected_output = """Specify the current trend stock price - up, down or sideways.
    eg. stock= 'APPL, price UP'
    """
)

#Importando a Tool Search
search_tool = DuckDuckGoSearchResults(
    backend = 'news', 
    num_results = 10
)

newsAnalyst = Agent(
    role =  "Stock News Analyst",
    goal = """Create a short summary of the market news related to the stock {ticket} company.
    Specify the current trend - up, down or sideways with the news context. For each request stock asset,
    specify a number between 0 and 100, where 0 is extreme fear an 100 is extreme greed.""",
    backstory = """You're a highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.
    
    You're also level analyts in tyradicional markets and have deep undersanding of human psychology.
    You understand news, theirs titles and informations, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles.""",
    verbose = True,
    llm = llm,
    max_iter = 10,
    memory = True,
    allow_delegation = False,
    tools = [search_tool]
)

getNews = Task(
    agent = newsAnalyst,
    description = f"""Take the stock always include BTC to it (if not request).
    Use the search tool to search each one individually
    The current date is {datetime.now()}.
    Compose the results intom a helpfuul report.""",
    expected_output = """A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTIONS>
    <FEAR/GREED SCORE>
    """
)

stockAnalystWrite = Agent(
    role =  "Senior Stock Writer",
    goal = """Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the 
    stock report and price trend.""",
    backstory = """You're a widely accepted as the best stock analyst in the market.
    You understand complex concepts and create compelling stories and narratives that 
    resonate with wider audiences.
    
    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multple opinions when analyzing anything.""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True
)

writeAnalyst = Task(
    agent = stockAnalystWrite,
    description = """Use the stock price trend and the stock news report to create an analyses and
    write the newsletter about the {ticket} company that is brief and highlights the most points.
    Focus on the stock price trend, news and fear/greed score, What are the near future  considerarions?
    Include the previous analyses of the stock trend and news summary.""",
    expected_output = """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. 
    It should cantain:
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up interest
    - main part provides the meat of the analysis ihncluding yhe news summary and fear/greed scores
    - summary - key facts and concrete future trend prediction - up , down or sideways.""",
    context = [getStockPrice, getNews]
)

crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, getNews, writeAnalyst],
    verbose = 2,
    process = Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15
)



with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the Ticket")
        submit_button = st.form_submit_button(label= "Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs ={'ticket': topic})

        st.subheader("Results of your Research")

        st.write(results['final_output'])