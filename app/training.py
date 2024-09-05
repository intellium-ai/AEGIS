import logging
from enum import Enum
from pathlib import Path
import pandas as pd

import streamlit as st
from components import Simulation, Agent
from streamlit import session_state as state

training_config_root = Path("../agents/training_configs/")
lay_down_config_root = Path("../data/laydown_configs/")

# NOTE: TRAINING BREAKS FOR EVERY AGENT -> SEEMINGLY MANY DIFFERENT REASONS?????

class TrainingStage(Enum):
    NOT_LOADED = 0
    READY = 1
    RUNNING = 2
    DONE = 3

if "agent" not in state:
    state.agent = None

if "laydown" not in state:
    state.laydown = None

if "stage" not in state:
    state.stage = TrainingStage.NOT_LOADED

st.markdown(
    """
    <style>
    .top-gap {
        margin-top: 32px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def update_stage():
    if state.laydown and state.agent:
        state.stage = TrainingStage.READY
    else:
        state.stage = TrainingStage.NOT_LOADED

def selection_component():
    # read CSV files
    agents_df = pd.read_csv('./metadata/agents.csv')
    laydowns_df = pd.read_csv('./metadata/laydowns.csv')

    trainable_agents = agents_df[agents_df['is_trainable'] == True]
    
    selected_agent = st.sidebar.selectbox(
        "Select an agent:",
        key="agents",
        options=trainable_agents['label'].tolist(),
        format_func=lambda x: x
    )

    if selected_agent:
        state.agent = trainable_agents[trainable_agents['label'] == selected_agent].iloc[0].to_dict()
        training_config_path = training_config_root / f"{state.agent['name']}.yaml"
    else:
        st.sidebar.write("Please select an agent")

    selected_laydown = st.sidebar.selectbox(
        "Select a laydown:",                
        key="laydowns",
        options=laydowns_df['label'].tolist(),
        format_func=lambda x: x
    )

    if selected_laydown:
        state.laydown = laydowns_df[laydowns_df['label'] == selected_laydown].iloc[0].to_dict()
        lay_down_config_path = lay_down_config_root / f"{state.laydown['name']}.yaml"
    else:
        st.sidebar.write("Please select a laydown")

    update_stage()

    st.markdown('<div class="top-gap" />', unsafe_allow_html=True)
    
    train_button_text = "Restart Training" if state.stage == TrainingStage.DONE else "Start Training"
    train_button = st.button(train_button_text, type="primary", disabled=(state.stage != TrainingStage.READY), use_container_width=True)

    agentObj = Agent(
        agent_class=state.agent['name'],         
        training_config_path=training_config_path,
        lay_down_config_path=lay_down_config_path
    )

    # this doesn't work
    if train_button:
        state.stage = TrainingStage.RUNNING
        agentObj.agent.learn()


    # for development debugging
    st.sidebar.write(f"Training Config Path: {training_config_path}")
    st.sidebar.write(f"Training Config Path: {lay_down_config_path}")



with st.sidebar:
    selection_component()

header_col_1, _, header_col_2 = st.columns([3, 1, 3], gap="small")

with header_col_1:
    st.title("Train Agent")
    st.write(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse at purus leo. Donec mollis eros a commodo eleifend. Pellentesque imperdiet, sapien vel consequat semper,"
    )

with header_col_2:
    # session_files = [path.name for path in session_config_root.iterdir()]

    col1, col2 = st.columns([1, 1])
    # with col1:
    #     st.write("**Agent:**")
    #     if state.simulation is not None:
    #         agent = state.simulation.agent
    #         training_config = agent._training_config
    #         agent_name = f"{training_config.agent_framework}: {training_config.agent_identifier}"
    #         st.markdown(f"{agent_name}")
    #     else:
    #         st.write("None selected.")
    #     st.write("")


st.divider()