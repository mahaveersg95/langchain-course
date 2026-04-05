import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # go up to parent

from utils.ssl_fix import apply_ssl_fix, get_http_client   
apply_ssl_fix()                                                

from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv(find_dotenv())

MAX_ITERATIONS = 10

@tool
def get_product_Price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop":45000, "mobile":25000, "headphones": 1500}
    return prices.get(product,0)

@tool
def apply_discount(price: float, usertype: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    discount_precent = {"bronze":10, "silver":20, "gold": 30}
    user_discount = discount_precent.get(usertype,0)
    return round(price * (1 - user_discount/100), 2)

@traceable(name="langchain Agent loop")
def run_agent(query: str):
    tools = [get_product_Price,apply_discount]
    tools_dic = {tool.name: tool for tool in tools} 

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=query)
    ]

    with get_http_client() as http_client: 
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, http_client=http_client)
        llm_with_tools = llm.bind_tools(tools)

        for iteration in range(1, MAX_ITERATIONS+1):
            print(f"\n--- Iteration {iteration} ---")

            ai_message = llm_with_tools.invoke(messages)
            tool_calls = ai_message.tool_calls

            # If no tool calls, this is the final answer
            if not tool_calls:
                print(f"\nFinal Answer: {ai_message.content}")
                return ai_message.content
            
            # Process only the FIRST tool call — force one tool per iteration
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args",{})
            tool_id = tool_call.get("id")

            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

            t_function = tools_dic.get(tool_name)

            if t_function is None:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            observation = t_function.invoke(tool_args)
            print(f"  [Tool Result] {observation}")

            messages.append(ai_message)
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))

    print("ERROR: Max iterations reached without a final answer")
    return None



if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    result = run_agent("What is the price of a headphones after applying a bronze discount?")
