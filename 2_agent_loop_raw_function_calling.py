from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "llama3.2:latest"

# --- Tools (Langchain @tåool decorator) ---

@traceable(run_type="tool", name="get_product_price")
def get_product_price(product : str) -> float:
    """ Look up the price of a product in the catalog. """
    print(f"  >>Executing get_product_price (product='{product}')")
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product, 0.0)

@traceable(run_type="tool", name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount to a price based on the discount tier. """
    print(f"  >>Executing apply_discount (price={price}, discount_tier='{discount_tier}')")
    discounts = {"silver": 12, "gold": 18, "platinum": 25}
    discount = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)

# Difference 2 : Without @tool, we must manually define the JSON schema for each function.
# This is exactly what Langchain's @tool decorator generate automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(run_type="llm", name="Ollama Chat")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)

# ---- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = { "get_product_price": get_product_price, "apply_discount": apply_discount }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {"role":"system", "content": """
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
        """},
        {"role":"user", "content": question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages)
        ai_message = response.message


        tool_calls = ai_message.tool_calls

        # if no tool call, this is the final answer.
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]

        #Difference 6: Attribute access.(.function.name) instead of dict access(.get("name"))

        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

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
        
        # Ensure numeric types are correctly cast if the LLM provides them as strings
        for key, value in normalized_args.items():
            if key == 'price':
                try:
                    normalized_args[key] = float(value)
                except (ValueError, TypeError):
                    pass

        observation = tool_to_use(**normalized_args)
        print(f"  [Tool Observation] {observation}")

        # To memorize the earlier observations and tool calls, we append the AI message and the tool observation to the messages list.
        messages.append(ai_message)
        messages.append(
            {"role": "tool", "content": str(observation), "name": tool_name}
        )



if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")