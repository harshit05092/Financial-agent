from typing import Annotated, List, Optional
from typing_extensions import TypedDict
class AgentState(TypedDict):
  messages : Annotated[List[str], add_messages]
  analyst_data : str
  tax_strategies : Annotated[List[str], add_messages]
  next_step : str