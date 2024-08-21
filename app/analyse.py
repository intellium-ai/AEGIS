import logging
from enum import Enum
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from components import Simulation
from streamlit import session_state as state

lay_down_config_root = Path("../data/laydown_configs/")
training_config_root = Path("../agents/training_configs/")
session_config_root = Path("../agents/trained_agents/")


# ----------------- STATE
if "simulation" not in state:
    state.simulation = None

if "slider_idx" not in state:
    state.slider_idx = 0

if "selected_node_idx" not in state:
    state.selected_node_idx = 0

if "selected_edge_idx" not in state:
    state.selected_edge_idx = 0
    
simulation = state.simulation

# ----------------- FUNCTIONS
def update_slider_idx(new_value):
    state.slider_idx = new_value
    

# ----------------- STYLES
st.markdown(
    """
    <style>
    .small-header {
        font-size: 16px;
        color: #FF4B4B;
        margin-top: 64px;
        margin-bottom: 0;
    }
    .justify-between {
        justify-content: space-between;
    }
    .justify-center {
        justify-content: center;
    }
    .centered-content {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    .value-container {
        display: flex;
        max-width: 1024px;
        gap: 64px;
        margin-top: 20px;
    }
    .observations {
        justify-content: start;
        margin-bottom: 8px;
    }
    .value-box {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .step {            
        width: 72px;
    }
    .pee {
        width: "max-content";
    }
    .reward {
        width: 180px;
    }
    .value-label {
        font-size: 16px;
        opacity: 0.5;
        margin-bottom: 0;
    }
    .value-text {
        font-size: 48px;
        margin-top: 0;
    }
    .changes-label {
        font-size: 16px;
        opacity: 0.5;
        margin-top: 20px;
    }
    .centered-content {
        display: flex;
        width: "100%";
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .margin-bottom {
        margin-bottom: 48px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- HEADER SECTION
header_col_1, _, header_col_2 = st.columns([3, 1, 3], gap="small")

with header_col_1:
    st.title("Post-Evaluation Analysis")
    st.write(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean sed neque malesuada, consectetur velit in, porttitor nisi. Nunc tincidunt in turpis a malesuada. "
    )

with header_col_2:
    pass

st.divider()

# ----------------- YOU SHALL NOT PASS SECTION
if not simulation:

    st.warning("""
        ### Hey hey hey...

        You can only use this page after evaluating a model!

        * **Step 1:** Head to the *Evaluation* page (hint: it's in the left navigation bar)
        * **Step 2:** Config an environment and click *Start Simulation*
        * **Step 3:** Come back here once your simulation has finished xx
    """)
    st.stop()
    

# easier to define some stuff here
node_name_list = simulation.history[0].nodes_table['Name'].tolist()
edge_name_list = simulation.history[0].traffic_table['Name'].tolist()

active_node = node_name_list[state.selected_node_idx]
active_edge = edge_name_list[state.selected_edge_idx]

# ----------------- OVERVIEW SECTION
with st.container():
    agent = state.simulation.agent
    training_config = agent._training_config
    agent_name = f"{training_config.agent_framework}: {training_config.agent_identifier}"

    st.markdown(
        f"""
        <div class="centered-content">
            <p class="small-header">SIMULATION OVERVIEW</p>            
            <div class="value-container justify-between">
                <div class="value-box pee">
                    <p class="value-label">Agent</p>
                    <p class="value-text">{agent_name}</p>
                </div>
                <div class="value-box step">
                    <p class="value-label">Steps</p>
                    <p class="value-text">{len(state.simulation.history) - 1}</p>
                </div>
                <div class="value-box reward">
                    <p class="value-label">Final Reward (avg)</p>
                    <p class="value-text">{state.simulation.avg_reward}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True)
    

# ----------------- GRAPH SECTION
with st.container():
    st.markdown(
        f"""
        <div class="centered-content margin-bottom">
            <p class="small-header">GRAPHS + STUFF</p>            
        </div>
        """,
        unsafe_allow_html=True
    )

    reward_graph_col_1, _, reward_graph_col_2 = st.columns([3, 1, 3], gap="small")
    feature_graph_col_1, _, feature_graph_col_2 = st.columns([3, 1, 3], gap="small")

    # calculate the cumulative rewards and average rewards at each step
    rewards = [state.reward if state.reward is not None else 0 for state in simulation.history]
    steps = np.arange(1, len(simulation.history) + 1)
    cumulative_rewards = np.cumsum(rewards)
    average_rewards = cumulative_rewards / steps

    chart_data_1 = pd.DataFrame({
        "Step": steps,
        "Reward": rewards,
        "Average Reward": average_rewards
    })

    chart_data_2 = pd.DataFrame({
        "Step": steps,
        "Cumulative Reward": cumulative_rewards,
    })

    reward_graph_col_1.line_chart(chart_data_1.set_index("Step"), y=["Reward", "Average Reward"])
    reward_graph_col_2.line_chart(chart_data_2.set_index("Step"), y=["Cumulative Reward"])

    # init empty lists
    node_data, edge_data = [], []

    # collect data for each step from index 0 to 100
    for step_idx in range(min(101, len(simulation.history))):
        nodes_table = simulation.history[step_idx].nodes_table
        edges_table = simulation.history[step_idx].traffic_table
        nodes_table['Step'] = step_idx 
        edges_table['Step'] = step_idx 
        node_data.append(nodes_table)
        edge_data.append(edges_table)

    # concatenate all the data into a single DataFrame
    full_node_data_df = pd.concat(node_data, ignore_index=True)   
    full_edge_data_df = pd.concat(edge_data, ignore_index=True)

    # create a list of unique node names
    node_name_list = full_node_data_df['Name'].unique()
    edge_name_list = full_edge_data_df['Name'].unique()

    # allow the user to select a node
    active_node = feature_graph_col_1.selectbox("Choose Node", node_name_list)
    active_edge = feature_graph_col_2.selectbox("Choose Edge", edge_name_list)

    # filter the DataFrame based on the selected node
    filtered_node_df = full_node_data_df[full_node_data_df['Name'] == active_node]
    filtered_edge_df = full_edge_data_df[full_edge_data_df['Name'] == active_edge]

    # drop the 'Name' column to display only the desired values
    desired_node_values = filtered_node_df.drop('Name', axis=1)
    desired_edge_values = filtered_edge_df.drop('Name', axis=1)

    # create charts
    feature_graph_col_1.write(desired_node_values) # TODO: Decide what to show for node features over time
    feature_graph_col_2.line_chart(desired_edge_values.set_index("Step"), y=["TCP Traffic", "TCP_SQL Traffic", "UDP Traffic"])


# ----------------- STEP-BY-STEP SECTION

with st.container():
    st.markdown(
        f"""
        <div class="centered-content">
            <p class="small-header">STEP BY STEP</p>            
        </div>
        """,
        unsafe_allow_html=True)
    
    slider_placeholder = st.empty()  # Placeholder for slider
    slider_value = slider_placeholder.slider(
        "Select a value",
        min_value=0,
        max_value=len(state.simulation.history) - 1,
        value=state.slider_idx,
        on_change=update_slider_idx,
        args=(state.slider_idx,)
    )

    active_state = state.simulation.history[slider_value]

    # TODO: fix the flashing when repopulating the graphic 
    main_col_1, main_col_2 = st.columns([1, 1], gap="large")
    table_col_1, table_col_2 = st.columns([2, 1], gap="large")
    node_table_label_placeholder = table_col_1.empty()
    node_table_placeholder = table_col_1.empty()
    traffic_table_label_placeholder = table_col_2.empty()
    traffic_table_placeholder = table_col_2.empty()
    
    with main_col_1:
        env_view = st.empty() 
        env_view.pyplot(active_state.network_figure, clear_figure=False)
    
    with main_col_2:
        values_placeholder = st.empty()  
        changes_placeholder = st.empty()
        changes_content_placeholder = st.empty()

        values_placeholder.markdown(
            f"""
            <div class="value-container justify-center">
                <div class="value-box step">
                    <p class="value-label">Step</p>
                    <p class="value-text">{slider_value}</p>
                </div>
                <div class="value-box reward">
                    <p class="value-label">Reward</p>
                    <p class="value-text">{active_state.reward}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        changes_placeholder.markdown(
            f"""
            <div class="value-container observations">
                <p class="value-label changes-label">Observation Space Changes</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with changes_content_placeholder.container():
            changes = active_state.changes
            if changes:
                for change in changes:
                    st.markdown(change)
            else:
                st.markdown("<p>No changes</p>", unsafe_allow_html=True)

    # Tables
    node_table_label_placeholder.markdown(
        f"""
        <div class="value-container observations">
            <p class="value-label changes-label">Nodes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    traffic_table_label_placeholder.markdown(
        f"""
        <div class="value-container observations">
            <p class="value-label changes-label">Traffic</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    node_table_placeholder.write(active_state.nodes_table)
    traffic_table_placeholder.write(active_state.traffic_table)