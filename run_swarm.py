from matplotlib import pyplot as plt
# plt.switch_backend('agg')

# Load the environment
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append("/home/yemika/Mikael/Work/UIUC/Projects/AgenticRobots/crazyswarm/ros_ws/src/crazyswarm/scripts")
from pycrazyswarm import Crazyswarm

import os
import pickle
import requests
import numpy as np
from typing import List
from functools import partial
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# Save the messages
def save_messages(messages, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(messages, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    Z = 1.0
    TAKEOFF_DURATION = 2.5
    GOTO_DURATION = 3.0
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    cf.takeoff(targetHeight=Z, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)

    # Configs
    SAVE_DIR = "experiments"
    EXPERIMENT_NAME = "test"

    # Create directories
    os.makedirs(os.path.join(SAVE_DIR, EXPERIMENT_NAME), exist_ok=True)

    def navigate_drone_tool(target_coordinate: List[float]):
        """Navigate a drone to target_coordinate specified by the input.
        
        Args:
            target_coordinate: target coordinate in the format (x, y, z).
        """
        target = np.array(target_coordinate)
        cf.goTo(target, yaw=0.0, duration=3)
        timeHelper.sleep(GOTO_DURATION + 1.0)
        return "Status: OK"

    def land_drone_tool():
        """Land the drone from the current position."""
        cf.land(targetHeight=0.05, duration=TAKEOFF_DURATION)
        timeHelper.sleep(TAKEOFF_DURATION + 1.0)
        return "Status: OK"
    
    def get_drone_position_tool():
        """Returns the drone's last measured position."""
        return cf.position()

    # Instantiate the list of tools
    tools_list = [
        navigate_drone_tool,
        land_drone_tool,
        get_drone_position_tool
    ]

    # System message
    system_message = """
    You are a helpful assistant who controls a flying robot. 
    Make sure you always fly at 1 meter height, unless explicitly told otherwise. 
    The room has the following dimensions: x_min=-2, x_max=+2, y_min=-2, y_max=+2.
    You can perform the following actions with the robot:
    - navigate the robot to a waypoint
    - land the robot
    - obtain the robot's current position
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
    
    # Save the conversation
    save_path = os.path.join(SAVE_DIR, EXPERIMENT_NAME, "conversation.pkl")
    try:
        save_messages(result['messages'], save_path)
    except Exception as e:
        print(f"Error saving the conversation:\n{e}")
    print(f"Conversation has been successfully saved to: {save_path}")

if __name__ == "__main__":
    main()