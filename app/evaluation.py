import logging
from enum import Enum
from pathlib import Path

import streamlit as st
from components import Simulation
from streamlit import session_state as state


class EvalStage(Enum):
    NOT_LOADED = 0
    READY = 1
    RUNNING = 2
    DONE = 3


lay_down_config_root = Path("../data/laydown_configs/")
training_config_root = Path("../agents/training_configs/")
session_config_root = Path("../agents/trained_agents/")

if "simulation" not in state:
    state.simulation = None

if "stage" not in state:
    state.stage = EvalStage.NOT_LOADED


def init_simulation():
    if state.session_file is not None:
        session_path = session_config_root / state.session_file
        state.simulation = Simulation.from_session_file(session_path=session_path)
        state.stage = EvalStage.READY
    elif state.laydown_file is not None and state.training_file is not None:
        lay_down_config_path = lay_down_config_root / state.laydown_file
        training_config_path = training_config_root / state.training_file
        state.simulation = Simulation.from_laydown(
            laydown_path=lay_down_config_path, training_path=training_config_path
        )
        state.stage = EvalStage.READY
    else:
        state.simulation = None
        state.stage = EvalStage.NOT_LOADED


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
    st.write("**Choose either:**")
    st.write("A pre-trained RL agent")
    session_files = [path.name for path in session_config_root.iterdir()]
    st.selectbox(
        "Trained agents",
        options=session_files,
        index=None,
        on_change=init_simulation,
        key="session_file",
    )

    st.write("")
    st.write("Or a new lay-down and training config")
    laydown_files = [path.name for path in lay_down_config_root.iterdir()]
    st.selectbox(
        "Lay down config",
        options=laydown_files,
        index=None,
        on_change=init_simulation,
        key="laydown_file",
        disabled=(state.session_file is not None),
    )

    training_files = [path.name for path in training_config_root.iterdir()]

    st.selectbox(
        "Training config",
        options=training_files,
        index=training_files.index("do_nothing.yaml"),
        on_change=init_simulation,
        key="training_file",
        disabled=(state.session_file is not None),
    )


with st.sidebar:
    selection_component()

header_col_1, _, header_col_2 = st.columns([3, 1, 3], gap="small")

with header_col_1:
    st.title("Evaluate Agent")
    st.write(
        "Run a PrimAITE simulation with the selected configuration and see how the environment changes in real time, how the blue agent reacts to the changes and what reward it receives."
    )

with header_col_2:
    session_files = [path.name for path in session_config_root.iterdir()]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Agent:**")
        if state.simulation is not None:
            agent = state.simulation.agent
            training_config = agent._training_config
            agent_name = f"{training_config.agent_framework}: {training_config.agent_identifier}"
            st.markdown(f"{agent_name}")
        else:
            st.write("None selected.")
        st.write("")

        eval_button_text = "Restart Simulation" if state.stage == EvalStage.DONE else "Start Simulation"
        evaluate_button = st.button(eval_button_text, type="primary", disabled=(state.simulation is None))

        # Show slider to select env state
        if state.stage == EvalStage.DONE:
            assert isinstance(state.simulation, Simulation)
            curr_step = st.slider(
                "Current step",
                value=len(state.simulation.history) - 1,
                min_value=0,
                max_value=len(state.env_history) - 1,
            )


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
    .changes-label {
        font-size: 16px;
        opacity: 0.5;
        margin-top: 20px;
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
    node_table_placeholder.table(env_state.nodes_table)
    traffic_table_placeholder.table(env_state.traffic_table)


if state.simulation is not None:
    if evaluate_button:
        run_simulation()
        st.rerun()

    if state.stage == EvalStage.READY:
        fig = state.simulation.latest_state.network_figure
        env_view.pyplot(fig)  # initially populate
    elif state.stage == EvalStage.DONE:
        env_state = state.simulation.history[curr_step]
        render_state(env_state, curr_step, env_state.reward)

logging.info(f"Simulation loaded: {state.simulation is not None}")
logging.info(f"Simulation stage: {state.stage.name}")
