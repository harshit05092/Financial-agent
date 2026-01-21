from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Removed redundant sys.path.append and import statements here
from my_utils import CAFinancialAgent
class CAGraph:
  def __init__(self, tax_doc_path : str, OPENAI_API_KEY : str):

    self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    self.call_financial_agent = CAFinancialAgent(OPENAI_API_KEY=OPENAI_API_KEY)
    ## Initializing the vector databases to be used as out tools
    self.fill_user_balanced_sheet = self.call_financial_agent.upload_user_balance_sheet(tax_doc_path)
    # self.fill_income_tax_act = self.call_financial_agent.ingest_tax_act(folder_path_with_pdf)
    self.user_sheet_retriever = self.call_financial_agent.get_agent_tools()

    self.checkpointer = MemorySaver()
    ## Initializing the calculator tool
    # self.calculator_tool = Financial_tools.get_calculation_tool()

    self.graph = self._build_graph()
  
  def supervisor_node(self, state : AgentState):
    messages = state['messages']
    system_prompt = Prompt.supervisor_prompt()
    response = self.llm.with_structured_output(Router).invoke(
        [("system", system_prompt)] + messages
    )
    return {
        "messages": [AIMessage(content=f"Routing to: {response.next}")], 
        "next": response.next 
          }

  def analyst_node(self, state : AgentState):
    
        user_retriever = self.user_sheet_retriever
        last_message = state["messages"][-1].content
        search_query = self.llm.invoke(f"Generate a concise 3-5 word search query for financial data based on: '{last_message}'").content

        # 2. Invoke Retriever
        # NOTE: This line AUTOMATICALLY converts 'search_query' (string) -> Embeddings (Numbers)
        # using the OpenAIEmbeddings model defined inside 'user_retriever'.
        retrieved_docs = user_retriever.invoke(search_query)
        context_text = "\n\n".join([d.page_content for d in retrieved_docs])

        extracted_data = self.llm.with_structured_output(AnalystReport).invoke(
            [SystemMessage(content="Extract raw figures. Return 0 if not found."), HumanMessage(content=context_text)]
        )

        summary_text = f"Based on the retrieved segments, I found:\n"
        for sector in extracted_data.sector_breakdown:
          summary_text += f"- **{sector.sector_name}**: {sector.total_amount}\n"

        return {
        # Append the assistant's response to the message history
        "messages": [AIMessage(content=summary_text)],
        
        # Save the structured data to state (if your state has this field)
        # This allows other nodes (like a Reporter) to use the raw numbers later.
        "analyst_data": extracted_data.dict() 
        }

  def tax_researcher_node(self, state : AgentState):
    messages = state['messages']
    system_prompt = Prompt.tax_researcher_prompt()

    tavily_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced", # "advanced" is better for financial facts
    include_answer=True,
    include_raw_content=False
    )

    llm_with_tools = self.llm.bind_tools(tools=[tavily_tool])
    
    system_msg = SystemMessage(content="You are a helpful assistant...")
    all_messages = [system_msg] + state["messages"]

    result = llm_with_tools.invoke(all_messages)

    return {"messages": [result["messages"][-1]],
            "tax_strategies": [result["messages"][-1]]
            }
  
  def _build_graph(self):

    workflow = StateGraph(AgentState) 

    workflow.add_node("Supervisor", self.supervisor_node)
    workflow.add_node("Tax_Researcher", self.tax_researcher_node)
    workflow.add_node("Financial_Analyst", self.analyst_node)

    workflow.add_edge(START, "Supervisor")

    workflow.add_edge("Financial_Analyst", "Supervisor")
    workflow.add_edge("Tax_Researcher", "Supervisor")
    workflow.add_conditional_edges(
            "Supervisor",
            lambda state: state["next"],
            {
                "Tax_Researcher": "Tax_Researcher",
                "Financial_Analyst": "Financial_Analyst",
                "Finish" : END
            }
    )

    return workflow.compile(checkpointer=self.checkpointer)