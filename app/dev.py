from pathlib import Path

import streamlit as st
from environment import display_env_state, EnvironmentState
from streamlit import session_state as state

from primaite.primaite_session import AgentIdentifier, PrimaiteSession

st.set_page_config(layout="wide", page_title="Home")

config_path = Path("../src/primaite/config/_package_data/")
lay_down_config_root = config_path / "lay_down"
training_config_root = config_path / "training"
session_config_root = Path("trained_agents")

# initialise session state for navigation if it doesn't exist
if "page" not in st.session_state:
    st.session_state.page = "home"

if "agent" not in state:
    state.agent = None

if "simulation_done" not in state:
    state.simulation_done = False

if "env_history" not in state:
    state.env_history = []

if "total_reward" not in state:
    state.total_reward = 0

if "laydown_file" not in state:
    state.laydown_file = None

if "curr_step" not in state:
    state.curr_step = 0

if "training_file" not in state:
    state.training_file = None

if "session_file" not in state:
    state.session_file = None

if "network_fig" not in state:
    state.network_fig = None


def init_primaite():

    if state.session_file is not None:
        session_path = session_config_root / state.session_file
        session = PrimaiteSession(session_path=session_path)
        session.setup()

        state.agent = session._agent_session
        env = state.agent._env
        state.env_history = [EnvironmentState(env)]

        state.simulation_done = False

    if state.laydown_file is not None and state.training_file is not None:
        lay_down_config_path = lay_down_config_root / state.laydown_file
        training_config_path = training_config_root / state.training_file

        session = PrimaiteSession(training_config_path, lay_down_config_path)
        session.setup()

        state.agent = session._agent_session
        env = state.agent._env
        state.env_history = [EnvironmentState(env)]

        state.simulation_done = False

    state.network_fig = state.env_history[state.curr_step].display_network()


def home_page():
    col1, col2 = st.columns(2, gap="large", vertical_alignment="center")

    st.markdown(
        """
        <style>
        .spacer {
            margin-top: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with col1:
        st.title("AEGIS - The Autonomous Embedded-Graph Intrusion Sentinel")
        st.write(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer non elit quis augue blandit suscipit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas."
        )

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # Create a horizontal container for the buttons
        button_container = st.container()

        # Add buttons in separate columns within the container
        button_col1, button_col2, _ = button_container.columns([1, 1, 1])

        with button_col1:
            if st.button("Evaluate Agent", use_container_width=True, type="primary"):
                st.session_state.page = "evaluate"
                st.rerun()

        with button_col2:
            if st.button("Train Agent", use_container_width=True, type="secondary"):
                st.session_state.page = "train"
                st.rerun()

    # Column 2: Hero graphic
    with col2:
        st.image(use_column_width=True, image="../tower_defense.png")


def evaluate_page():
    with st.container():
        if st.button("⬅"):
            st.session_state.page = "home"
            st.experimental_rerun()

    header_col_1, _, header_col_2 = st.columns([3, 1, 3], gap="small")

    with header_col_1:
        st.title("Evaluate Agent")
        st.write(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer non elit quis augue blandit suscipit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas."
        )

    with header_col_2:
        session_files = [path.name for path in session_config_root.iterdir()]

        session_file = st.selectbox(
            label="Trained agents",
            label_visibility="hidden",
            options=session_files,
            index=None,
            on_change=init_primaite,
            key="session_file",
            placeholder="Select a trained agent...",
        )

        evaluate_button = st.button("Start Simulation", type="primary", disabled=False if session_file else True)

    if not session_file:
        return

    st.divider()

    main_col_1, main_col_2 = st.columns([1, 1], gap="large")
    table_col_1, table_col_2 = st.columns([2, 1], gap="large")

    env_view = main_col_1.empty()  # Create a placeholder for the network figure
    values_placeholder = main_col_2.empty()  # Placeholder for step and reward values
    changes_placeholder = main_col_2.empty()  # Placeholder for changes label
    changes_content_placeholder = main_col_2.empty()  # Placeholder for actual changes

    # Placeholders for tables with static labels
    node_table_label_placeholder = table_col_1.empty()
    node_table_placeholder = table_col_1.empty()

    traffic_table_label_placeholder = table_col_2.empty()
    traffic_table_placeholder = table_col_2.empty()

    env_view.pyplot(state.network_fig)  # initially populate

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

    if evaluate_button:
        agent = state.agent
        env = agent._env
        env.set_as_eval()
        obs = env.reset()
        prev_env_state = EnvironmentState(env)
        state.env_history = [prev_env_state]
        state.total_reward = 0
        num_steps = state.agent._training_config.num_eval_steps

        for step in range(1, num_steps + 1):
            # Run simulation
            prompt = None
            reasoning = None
            if env.agent_identifier == AgentIdentifier.LLM:
                action, prompt, reasoning = agent._calculate_action_info(obs)
            else:
                action, _, _ = agent._calculate_action_info(obs)
            obs, rewards, done, _ = env.step(action)
            state.total_reward += rewards

            env_state = EnvironmentState(env, prev_env_state, action, prompt=prompt, reasoning=reasoning)
            state.network_fig = env_state.display_network()

            state.env_history.append(env_state)

            prev_env_state = env_state

            env_view.pyplot(state.network_fig)  # Update the figure

            values_placeholder.markdown(
                f"""
                <div class="value-container">
                    <div class="value-box step">
                        <p class="value-label">Step</p>
                        <p class="value-text">{step}</p>
                    </div>
                    <div class="value-box reward">
                        <p class="value-label">Reward</p>
                        <p class="value-text">{round(state.total_reward / (len(state.env_history) - 1), 5)}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            changes_html = ""
            changes = env_state.obs_diff(colors=True)
            if changes:
                for change in changes:
                    changes_html += change
            else:
                changes_html = "<p>No changes</p>"

            changes_placeholder.markdown(
                f"""
                <div class="value-container observations">
                    <p class="value-label changes-label">Observation Space Changes</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            changes_content_placeholder.markdown(changes_html, unsafe_allow_html=True)

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

            if done:
                break


def train_page():
    st.title("Train Agent")
    st.write(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer non elit quis augue blandit suscipit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas."
    )

    if st.button("Go Back to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()


# navigation logic
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "evaluate":
    evaluate_page()
elif st.session_state.page == "train":
    train_page()
