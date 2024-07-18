# Interactive evaluation app for a primaite session

import glob
import logging
import os
import sys
from pathlib import Path
import shutil

import streamlit as st
from environment import display_env_state, EnvironmentState
from streamlit import session_state as state

from primaite.primaite_session import PrimaiteSession, AgentIdentifier

st.set_page_config(layout="wide")

config_path = Path("../src/primaite/config/_package_data/")
lay_down_config_root = config_path / "lay_down"
training_config_root = config_path / "training"
session_config_root = Path("trained_agents")

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


def cleanup_output():
    if state.session_file is not None:
        session_path = session_config_root / state.session_file
        shutil.rmtree(path=session_path / "evaluation")
        for p in session_path.glob("network_*"):
            p.unlink()


with st.sidebar:
    st.write("**Choose either:**")
    st.write("A pre-trained RL agent")
    session_files = [path.name for path in session_config_root.iterdir()]
    st.selectbox(
        "Trained agents",
        options=session_files,
        index=None,
        on_change=init_primaite,
        key="session_file",
    )

    st.write("")
    st.write("Or a new training and lay-down config")
    laydown_files = [path.name for path in lay_down_config_root.iterdir()]
    st.selectbox(
        "Lay down config",
        options=laydown_files,
        index=None,
        on_change=init_primaite,
        key="laydown_file",
    )

    training_files = [path.name for path in training_config_root.iterdir()]
    training_file = st.selectbox(
        "Training config",
        options=training_files,
        index=None,
        on_change=init_primaite,
        key="training_file",
    )

    if state.agent is not None:
        training_config = state.agent._training_config
        agent_name = f"{training_config.agent_framework}: {training_config.agent_identifier}"
        st.write("**Agent:**")
        st.write(f":yellow[{agent_name}]")

if state.agent is not None:
    input_col, curr_step_col, caption_col = st.columns([1, 2, 2])
    with input_col:
        button = st.button("Start PrimAITE simulation")

    with curr_step_col:
        if len(state.env_history) > 1:
            state.curr_step = st.slider(
                "Current step", value=len(state.env_history) - 1, min_value=0, max_value=len(state.env_history) - 1
            )
        else:
            state.curr_step = 0

    with caption_col:
        if state.simulation_done:
            st.write(f"Avg Reward for Episode: :orange[{round(state.total_reward / (len(state.env_history) - 1), 5)}]")
    st.divider()
    env_view = st.empty()

    # Run the simulation
    if button:
        agent = state.agent
        env = agent._env
        env.set_as_eval()
        obs = env.reset()
        prev_env_state = EnvironmentState(env)
        state.env_history = [prev_env_state]
        state.total_reward = 0
        num_steps = state.agent._training_config.num_eval_steps

        for step in range(1, num_steps + 1):
            env_view.empty()

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
            with env_view.container():
                st.markdown(f"Step :orange[{step}]&emsp; Avg Reward: :orange[{round(state.total_reward / step, 5)}]")
                display_env_state(env_state)
            state.env_history.append(env_state)

            prev_env_state = env_state

            if done:
                break
        state.simulation_done = True
        cleanup_output()

        st.rerun()
    else:
        env_view.empty()
        with env_view.container():
            display_env_state(state.env_history[state.curr_step])
