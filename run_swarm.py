# Load the environment
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append("/home/yemika/Mikael/Work/UIUC/Projects/AgenticRobots/crazyswarm/ros_ws/src/crazyswarm/scripts")
from pycrazyswarm import Crazyswarm

import requests
import numpy as np
from matplotlib import pyplot as plt
# plt.switch_backend('agg')
from typing import List
from functools import partial
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# Define some tools
def weather_tool(location: str):
    """
    Fetches the current weather information for a specified location using the OpenWeatherMap API.

    Args:
        location: The name of the location (city, region, or country) for which to retrieve the weather data. Must be a complete name, not shortened.

    Returns:
        dict: A dictionary containing weather information, including:
            - location (str): The name of the location.
            - temperature (float): The current temperature in Celsius.
            - description (str): A brief description of the current weather (e.g., "clear sky").
            - humidity (int): The current humidity percentage.
            - wind_speed (float): The wind speed in meters per second.
        
        str: An error message if the weather data could not be retrieved.
        
    Notes:
        - The function uses a hardcoded API key. Ensure that it is kept secure and updated as needed.
        - The temperature is returned in Celsius. Use 'imperial' units for Fahrenheit if needed by modifying the `units` parameter.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': location,
        'appid': "aa200e6b28720e9cfca17f9af0e73f35",
        'units': 'metric'  # Use 'imperial' for Fahrenheit
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        weather = {
            'location': data['name'],
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
        return weather
    else:
        return f"Could not retrieve the weather. Response status: {response.status_code}."

def multiply_tool(a: float, b: float) -> float:
    """Multiply a and b.

    Args:
        a: first float
        b: second float
    """
    return a * b

def round_tool(a: float, b: int) -> float:
    """Round float a to b digits.

    Args:
        a: number that needs to be rounded
        b: number of digits
    """
    return round(a, b)

# def navigate_drone_tool(target_coordinate: List[float]):
#     """Navigate a drone to target_coordinate specified by the input.
    
#     Args:
#         target_coordinate: target coordinate in the format (x, y, z).
#     """
#     # global cf
#     # global GOTO_DURATION
#     # global timeHelper
#     target = cf.initialPosition + np.array(target_coordinate)
#     cf.goTo(target, yaw=0.0, duration=3)
#     timeHelper.sleep(GOTO_DURATION + 1.0)

# Function to build and compile the graph
def init_graph(tools_list: List, system_message: str):
    # Initialize the LLM model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Initialize the graph
    builder = StateGraph(MessagesState)

    # Initialize the tools list
    llm_with_tools = llm.bind_tools(tools_list)

    # Specify the system message
    sys_message = SystemMessage(content=system_message)

    # Define the LLM node
    def llm_node(state: MessagesState):
        response = llm_with_tools.invoke(
            [sys_message] + state['messages']
        )
        state['messages'].append(response)

    # Add the nodes
    builder.add_node("assistant", llm_node)
    builder.add_node("tools", ToolNode(tools_list))

    # Add the edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    builder.add_edge("tools", "assistant")

    # Initialize the memory
    memory = MemorySaver()

    # Compile the graph
    graph = builder.compile(checkpointer=memory)

    return graph

def main():
    # Initialize the swarm simulation
    # global cf
    # global GOTO_DURATION
    # global timeHelper
    Z = 1.0
    TAKEOFF_DURATION = 2.5
    GOTO_DURATION = 3.0
    WAYPOINTS = np.array([
        (1.0, 0.0, Z),
        (1.0, 1.0, Z),
        (0.0, 1.0, Z),
        (0.0, 0.0, Z),
    ])
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    cf.takeoff(targetHeight=Z, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)

    def navigate_drone_tool(target_coordinate: List[float]):
        """Navigate a drone to target_coordinate specified by the input.
        
        Args:
            target_coordinate: target coordinate in the format (x, y, z).
        """
        # global cf
        # global GOTO_DURATION
        # global timeHelper
        target = cf.initialPosition + np.array(target_coordinate)
        cf.goTo(target, yaw=0.0, duration=3)
        timeHelper.sleep(GOTO_DURATION + 1.0)

    # Instantiate the list of tools
    tools_list = [
        multiply_tool,
        round_tool,
        weather_tool,
        navigate_drone_tool
    ]

    # System message
    system_message = """
    You are a helpful assistant who controls a flying robot. You can set the waypoints for the robot.
    """

    # Initialize the graph
    graph = init_graph(
        tools_list=tools_list, 
        system_message=system_message
    )

    # Initialize the memory configuration
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # Start a chat with the user
    while True:
        # Get the user input
        user_input = input("User: ")

        # Check if exit requested
        if user_input in ['exit', 'quit']:
            print("Exiting the chat. Bye!")
            break

        # Otherwise, respond to the user
        user_message = HumanMessage(content=user_input)
        user_messages = [user_message]

        # Get the model's response
        result = graph.invoke(
            {"messages": user_messages},
            config
        )

        # Print the LLM's response
        for m in result['messages']:
            m.pretty_print()
        
        # Navigate the drone
        cf.goTo(WAYPOINTS[0], yaw=0.0, duration=GOTO_DURATION)
        
        
        # navigate_drone_tool([2, 2, 2])

if __name__ == "__main__":
    main()