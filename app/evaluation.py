import logging
from enum import Enum
from pathlib import Path
import pickle
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit import session_state as state
from primaite.common.simulation import Simulation


class EvalStage(Enum):
    NOT_LOADED = 0
    READY = 1
    RUNNING = 2
    DONE = 3

trained_agents_root = Path("../agents/trained_agents/")
lay_down_config_root = Path("../data/laydown_configs/")
training_config_root = Path("../agents/training_configs/")
session_config_root = Path("../agents/trained_agents/")
session_obj_save_root = Path("../data/evaluations")

if "use_trained_agent" not in state:
    state.use_trained_agent = None

if "simulation" not in state:
    state.simulation = None

if "stage" not in state:
    state.stage = EvalStage.NOT_LOADED

if "agent" not in state:
    state.agent = None

if "laydown" not in state:
    state.laydown = None

if "trained_agent_config" not in state:
    state.trained_agent_config = None

if "save_evaluation" not in state:
    state.save_evaluation = True

if "evaluation_data" not in state:
    state.evaluation_data = None

def init_simulation():

    # using a trained agent
    if state.trained_agent_config is not None:
        session_path = trained_agents_root / f"{state.trained_agent_config['name']}"
        state.simulation = Simulation.from_session_file(session_path=session_path)
        state.stage = EvalStage.READY

    # using an untrained agent
    elif state.laydown is not None and state.agent is not None:
        lay_down_config_path = lay_down_config_root / f"{state.laydown['name']}.yaml"
        training_config_path = training_config_root / f"{state.agent['name']}.yaml"
        state.simulation = Simulation.from_laydown(
            laydown_path=lay_down_config_path, training_path=training_config_path
        )
        state.stage = EvalStage.READY
    else:
        state.simulation = None
        state.stage = EvalStage.NOT_LOADED

@st.dialog("Simulation Complete")
def view_analysis_dialog():
    st.write("Click here for a step-by-step analysis of this simulation")
    if st.button("Analyse", type="primary"):
        st.write("Redirecting to Analyse Page...")
        st.switch_page("analyse.py")


def run_simulation():
    assert isinstance(state.simulation, Simulation)
    sim = state.simulation
    sim.reset()

    state.stage = EvalStage.RUNNING

    while not sim.is_done():
        sim.step()
        render_state(sim.latest_state, sim.curr_step, sim.avg_reward)

    state.stage = EvalStage.DONE

def selection_component():

    init_simulation()

    state.use_trained_agent = st.sidebar.toggle(label="Trained Agent", value=state.use_trained_agent)

    if state.use_trained_agent:
        st.write("Select an agent which has already been trained.\nUse the *Training* page to train agents.")
    else:
        st.write("Select a non-trainable agent and a laydown to evaluate.")

    trainings_df = pd.read_csv('./metadata/trainings.csv')
    trained_agents = trainings_df['agent'].unique().tolist() if not trainings_df.empty else []
    agents_df = pd.read_csv('./metadata/agents.csv')
    untrainable_agents = agents_df[agents_df['is_trainable'] == False]
    laydowns_df = pd.read_csv('./metadata/laydowns.csv')
    trainable_agents = agents_df[
        (agents_df['is_trainable'] == True) & 
        (agents_df['name'].isin(trained_agents))
    ]

    if state.use_trained_agent:

        selected_agent = st.sidebar.selectbox(
            "Trainable agent:",
            key="trainable_agents",
            options=trainable_agents['label'].tolist(),
            format_func=lambda x: x,
            on_change=init_simulation,
        )

        # config for pre-trained agent
        selected_training = st.sidebar.selectbox(
            "Training config:",                
            key="pretrained_laydowns",
            options=trainings_df['name'].tolist(),
            format_func=lambda x: x,
            on_change=init_simulation,
        )

        if selected_agent:
            state.agent = trainable_agents[trainable_agents['label'] == selected_agent].iloc[0].to_dict()
            training_config_path = training_config_root / f"{state.agent['name']}.yaml"
        else:
            st.sidebar.write("Please select an agent")

        if selected_training:
            state.trained_agent_config = trainings_df[trainings_df['name'] == selected_training].iloc[0].to_dict()
            trained_agent_config_path = trained_agents_root / f"{state.trained_agent_config['name']}.yaml"
        else:
            st.sidebar.write("Please select a training config")

        # for debug
        st.write(f"state.trained_agent_config path: \n{trained_agent_config_path}")

    else:
        # non-trained agent
        selected_agent = st.sidebar.selectbox(
            "Non-trainable agent:",
            key="untrainable_agents",
            options=untrainable_agents['label'].tolist(),
            format_func=lambda x: x,
            on_change=init_simulation,
        )

        if selected_agent:
            state.agent = untrainable_agents[untrainable_agents['label'] == selected_agent].iloc[0].to_dict()
            training_config_path = training_config_root / f"{state.agent['name']}.yaml"
        else:
            st.sidebar.write("Please select an agent")


        # laydown for non-trained agent
        selected_laydown = st.sidebar.selectbox(
            "Laydown:",                
            key="all_laydowns",
            options=laydowns_df['label'].tolist(),
            format_func=lambda x: x,
            on_change=init_simulation,
        )

        if selected_laydown:
            state.laydown = laydowns_df[laydowns_df['label'] == selected_laydown].iloc[0].to_dict()
            lay_down_config_path = lay_down_config_root / f"{state.laydown['name']}.yaml"
        else:
            st.sidebar.write("Please select a laydown")

        # for debug
        st.write(f"state.laydown path: \n{lay_down_config_path}")

    if state.use_trained_agent:
        state.laydown = None
    else:
        state.trained_agent_config = None

    # for debug
    st.write(f"state.agent path: \n{training_config_path}")

    st.markdown('<div class="mt" />', unsafe_allow_html=True)

    state.save_evaluation = st.sidebar.toggle(label="Save Evaluation", value=state.save_evaluation)


with st.sidebar:
    selection_component()


# ----------------------------------- HEADING

header_col_1, _, header_col_2 = st.columns([3, 1, 3], gap="small", vertical_alignment="bottom")

with header_col_1:
    st.title("Evaluate Agent")
    st.write(
        "Run a PrimAITE simulation with the selected configuration and see how the environment changes in real time, how the blue agent reacts to the changes and what reward it receives."
    )

with header_col_2:

    _, button_col = st.columns([2, 2])
    with button_col:

        button_disabled = False
        if not state.agent:
            button_disabled = True
        if not state.trained_agent_config and not state.laydown:
            button_disabled = True

        eval_button_text = "Restart Simulation" if state.stage == EvalStage.DONE else "Start Simulation"
        evaluate_button = st.button(eval_button_text, type="primary", use_container_width=True, disabled=(button_disabled))
        
        # if evaluate_button:
        #     run_simulation()


st.divider()

main_col_1, main_col_2 = st.columns([1, 1], gap="large")
table_col_1, table_col_2 = st.columns([2, 1], gap="large")

env_view = main_col_1.empty()  # Create a placeholder for the network figure
values_placeholder = main_col_2.empty()  # Placeholder for step and reward values
changes_placeholder = main_col_2.empty()  # Placeholder for changes label
changes_content_placeholder = main_col_2.empty()  # Placeholder for actual changes
action_label_placeholder = main_col_2.empty()
action_content_placeholder = main_col_2.empty()
info_label_placeholder = main_col_2.empty()
info_content_placeholder = main_col_2.empty()

# Placeholders for tables with static labels
node_table_label_placeholder = table_col_1.empty()
node_table_placeholder = table_col_1.empty()

traffic_table_label_placeholder = table_col_2.empty()
traffic_table_placeholder = table_col_2.empty()

st.markdown(
    """
    <style>
    .value-container {
        display: flex;
        justify-content: center;
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
    .mt {
        margin-top: 48px;
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
    </style>
    """,
    unsafe_allow_html=True,
)

def render_state(env_state, step, avg_reward):

    # Update the network figure
    env_view.pyplot(env_state.network_figure)

    rewards_delta = "" if not env_state.reward else f"({env_state.reward})"

    values_placeholder.markdown(
        f"""
        <div class="value-container">
            <div class="value-box step">
                <p class="value-label">Step</p>
                <p class="value-text">{step}</p>
            </div>
            <div class="value-box reward">
                <p class="value-label">Average reward</p>
                <p class="value-text">{avg_reward}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Observed Changes
    changes_placeholder.markdown(
        f"""
        <div class="value-container observations">
            <p class="value-label changes-label">Observation Space Changes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with changes_content_placeholder.container():
        changes = env_state.changes
        if changes:
            for change in changes:
                st.markdown(change)
        else:
            st.markdown("<p>No changes</p>", unsafe_allow_html=True)

    # Action and action info
    action = env_state.action

    action_label_placeholder.markdown(
        f"""
        <div class="value-container observations">
            <p class="value-label changes-label">Action</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action is not None:
        action_verbose = action.verbose(colored=True)
        action_content_placeholder.markdown(action_verbose)

    info_label_placeholder.markdown(
        f"""
        <div class="value-container observations">
            <p class="value-label changes-label">Info</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info = env_state.info
    if info is not None:
        info_content_placeholder.markdown(env_state.info_str)

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
    node_table_placeholder.write(env_state.nodes_table)
    traffic_table_placeholder.write(env_state.traffic_table)


if state.simulation is not None:
    if evaluate_button and state.stage != EvalStage.DONE:
        run_simulation()

    if state.stage == EvalStage.READY:
        fig = state.simulation.latest_state.network_figure
        env_view.pyplot(fig)  # initially populate

    if state.stage == EvalStage.DONE:

        evaluations_df = pd.read_csv('./metadata/evaluations.csv')
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        eval_obj = {
            'id': len(evaluations_df) + 1,
            'name': current_datetime,
            'agent': state.agent["name"],
            'training_config': state.trained_agent_config['name'] if state.trained_agent_config else None,
            'laydown': state.laydown["name"] if state.laydown else None,
            'final_reward_avg': state.simulation.avg_reward
        }

        # save session obj
        if state.save_evaluation:
            save_path = session_obj_save_root / f"{current_datetime}.pkl"
            new_row = pd.DataFrame([eval_obj])  # Wrap eval_obj in a list
            evaluations_df = pd.concat([evaluations_df, new_row], ignore_index=True)
            evaluations_df.to_csv('./metadata/evaluations.csv', index=False)
        
            with open(save_path, 'wb') as f:
                pickle.dump(state.simulation.history, f)

        state.evaluation_data = eval_obj
            
        view_analysis_dialog()
