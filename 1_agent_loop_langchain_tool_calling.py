from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "llama3.2:latest"

# --- Tools (Langchain @tool decorator) ---

@tool
def get_product_price(product : str) -> float:
    """ Look up the price of a product in the catalog. """
    print(f"  >>Executing get_product_price (product='{product}')")
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount to a price based on the discount tier. """
    print(f"  >>Executing apply_discount (price={price}, discount_tier='{discount_tier}')")
    discounts = {"silver": 12, "gold": 18, "platinum": 25}
    discount = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)


# ---- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(model = MODEL, model_provider = "ollama", temperature=0.0, base_url="http://localhost:11434")
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content="""
          You are a shopping assistant using a tool-based agent loop.
          You have exactly 2 tools to use in sequence.
          
          RULES:
          1. You MUST call get_product_price first with the product name
          2. After you receive the price, you MUST call apply_discount with that price and discount tier
          3. Do NOT provide any final answer until BOTH tools have been called and you have BOTH results
          4. Do NOT describe or format tool calls as text - actually invoke them using the tool interface
          5. Each iteration you make exactly ONE tool call
          
          For "What is the price of a laptop with a gold discount?":
          - Iteration 1: Call get_product_price with product="laptop"
          - Iteration 2: Call apply_discount with the price result and discount_tier="gold"
          - Iteration 3: Provide final answer with the discounted price
        """),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict.")
        
        # Normalize tool args: flatten nested dicts with 'type' key
        normalized_args = {}
        for key, value in tool_args.items():
            if isinstance(value, dict) and 'type' in value:
                normalized_args[key] = value['type']
            else:
                normalized_args[key] = value
        
        observation = tool_to_use.invoke(normalized_args)
        print(f"  [Tool Observation] {observation}")

        #  To memorize the earliar observations and tool calls, we append the AI message and the tool observation to the messages list.
        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )



if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")