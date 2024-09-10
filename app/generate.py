from enum import Enum
from pathlib import Path
import random

import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from streamlit import session_state as state
from torch_geometric.data import Batch
import torch

from primaite.agents.aegis.modules.openai import OpenAIClient
from primaite.network.generator import NODE_TYPE_MAP, NetworkGenerator, SERVICE_TO_PORT_MAP

load_dotenv() 
open_ai_key = os.getenv('OPEN_AI_KEY')

training_config_root = Path("../agents/training_configs/")
dataset_save_root = Path("../data/datasets") 

class GenerateStage(Enum):
    READY = 1
    RUNNING = 2
    DONE = 3

if 'is_dataset' not in state:
    state.is_dataset = False

if 'generation_output' not in state:
    state.generation_output = []

if 'generate_dataset_params' not in state:
    state.generate_dataset_params = {
        'dataset_name': None,
        'training_config_path': None,
        'seed': None,
        'size': 5,
        'number_of_nodes': (20, 30),
        'services': ["HTTP", "SSH"],
        'ports': ["80", "22"]
    }

if 'stage' not in state:
    state.stage = GenerateStage.READY

if 'progress' not in state:
    state.progress = 0

if 'progress_bar' not in state:
    state.progress_bar = st.sidebar.empty()

st.markdown(
    """
    <style>

    .centered-content {
        display: flex;
        width: "100%";
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .small-header {
        font-size: 16px;
        color: #FF4B4B;
        margin-bottom: 0;
    }

    .margin-top {
        margin-top: 64px;
    }

    .margin-bottom {
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def generate():
    if not state.generate_dataset_params["dataset_name"]:
        st.error("Please enter a dataset name.")
        return
    
    if not state.generate_dataset_params["training_config_path"]:
        st.error("Please select an agent to generate the network for.")
        return    
        
    if not state.generate_dataset_params["services"]:
        st.error("Please select at least 1 network protocol.")
        return
    
    try:
        state.stage = GenerateStage.RUNNING

        # if generating a dataset
        if state.is_dataset:
            total_items = state.generate_dataset_params["size"]
            for i in range(total_items):

                node_count = random.randint(state.generate_dataset_params["number_of_nodes"][0], state.generate_dataset_params["number_of_nodes"][1])


                generator = NetworkGenerator(
                    training_config_path=state.generate_dataset_params["training_config_path"],
                    graph_size=node_count,
                    random_seed=state.generate_dataset_params["seed"],
                    services=state.generate_dataset_params["services"],
                    ports=state.generate_dataset_params["ports"],
                    pretrained_llm=OpenAIClient(api_key=open_ai_key)
                )

                dataset_dir = dataset_save_root / f"{state.generate_dataset_params['dataset_name']}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                laydown_save_path = dataset_dir / f"{i}.yaml"

                output = generator.run_end_to_end(laydown_save_path=laydown_save_path)
                state.generation_output.append(output)

                progress_percentage = (i + 1) / total_items * 100
                state.progress = int(progress_percentage)
                state.progress_bar.progress(state.progress, text=f"Generating: {state.progress}%")


                print("progress_percentage:::", state.progress)



            try:
                # save to folder
                dataset_save_path = dataset_save_root / state.generate_dataset_params['dataset_name'] / "dataset.pt"
                batch = Batch.from_data_list(state.generation_output)
                torch.save(batch, dataset_save_path) 

            except:
                st.error("Oopsies... I couldn't save your dataset.")
                return
        # if just generating a laydown
        else:
            laydown_save_path = Path("../data") / "laydown_configs" / f"{state.generate_dataset_params['dataset_name']}.yaml"

            generator = NetworkGenerator(
                training_config_path=state.generate_dataset_params["training_config_path"],
                graph_size=state.generate_dataset_params["number_of_nodes"][0],
                random_seed=state.generate_dataset_params["seed"],
                services=state.generate_dataset_params["services"],
                ports=state.generate_dataset_params["ports"],
                pretrained_llm=OpenAIClient(api_key=open_ai_key)
            )

            # save laydown
            output = generator.run_end_to_end(laydown_save_path=laydown_save_path)

            laydowns_df = pd.read_csv('./metadata/laydowns.csv')

            # add a new row to the DataFrame
            new_row = pd.DataFrame({
                'id': [len(laydowns_df) + 1],
                'name': [state.generate_dataset_params['dataset_name']],
                'label': [state.generate_dataset_params['dataset_name']]
            })
            laydowns_df = pd.concat([laydowns_df, new_row], ignore_index=True)
            
            # save the updated DataFrame back to CSV
            laydowns_df.to_csv('./metadata/laydowns.csv', index=False)

            try:
                pass
            except:
                st.error("Oopsies... I couldn't save your laydown.")
                return
            
        state.progress = 0
        state.progress_bar.empty() 

        st.balloons()    
        
    except Exception as e:
        st.error(f"An error occurred while generating the network: {str(e)}")
        state.stage = GenerateStage.READY


state.is_dataset = st.sidebar.toggle(label="Generate Dataset", value=state.is_dataset)

if state.is_dataset:
    state.generate_dataset_params["size"] = st.sidebar.slider(label="Dataset Size", min_value=1, max_value=20, value=state.generate_dataset_params["size"])


# ----------------------------------- HEADING

header_col_1, _, header_col_2 = st.columns([3, 1, 3], gap="small", vertical_alignment="bottom")

with header_col_1:
    st.title("Generate Network")
    st.write(
        "Just configure a bunch of parameters and then generate a laydown (or dataset) ready for training + evaluating your agents"
    )

with header_col_2:
    _, button_col = st.columns([2, 2])
    with button_col:
        button_text = "Generate Dataset" if state.is_dataset else "Generate Laydown"
        generate_button = st.button(button_text, type="primary", use_container_width=True, disabled=state.stage == GenerateStage.RUNNING)
        if generate_button:
            generate()


st.divider()


# ----------------------------------- PARAMETER SELECTION

st.markdown(
    f"""
    <div class="margin-bottom">
        <p class="small-header">CONFIG</p>            
    </div>
    """,
    unsafe_allow_html=True
)  

config_params_col_1, _, config_params_col_2 = st.columns([3, 1, 3], gap="small")

with config_params_col_1:

    state.generate_dataset_params["dataset_name"] = st.text_input(label="Dataset name *" if state.is_dataset  else "Laydown name*", placeholder="Enter dataset name...", value=state.generate_dataset_params["dataset_name"])

    agents_df = pd.read_csv('./metadata/agents.csv')

    selected_agent = st.selectbox(
        "Agent:",
        key="agents",
        options=agents_df['label'].tolist(),
        format_func=lambda x: x
    )

    if selected_agent:
        state.agent = agents_df[agents_df['label'] == selected_agent].iloc[0].to_dict()
        state.generate_dataset_params["training_config_path"] = training_config_root / f"{state.agent['name']}.yaml"

    fix_seed = st.checkbox(label="Fix Seed", value=state.generate_dataset_params["seed"] is not None)
    state.generate_dataset_params["seed"] = 1729 if fix_seed else None


    st.markdown(
        f"""
        <div class="margin-bottom margin-top">
            <p class="small-header">GRAPHS</p>            
        </div>
        """,
        unsafe_allow_html=True
    )  
        
    if state.is_dataset:
        min_nodes, max_nodes = st.slider(
            label="Number of Nodes (Range)",
            min_value=5,
            max_value=100,
            value=(20, 30), 
            key="dataset_node_range"
        )
        state.generate_dataset_params["number_of_nodes"] = (min_nodes, max_nodes)
    else:
        single_slider = st.slider(
            label="Number of Nodes",
            min_value=5,
            max_value=100,
            value=20,
            key="single_node_count"
        )

        state.generate_dataset_params["number_of_nodes"] = (single_slider, single_slider)
    

    selected_services = st.multiselect(
        label="Network Protocols *",
        options=["HTTP", "SSH", "FTP"],
        default=["HTTP", "SSH"],
    )

    if selected_services:
        state.generate_dataset_params["services"] = selected_services
        state.generate_dataset_params["ports"] = [SERVICE_TO_PORT_MAP[service] for service in selected_services]


generator = NetworkGenerator(
    training_config_path=state.generate_dataset_params["training_config_path"], 
    graph_size=state.generate_dataset_params["number_of_nodes"][0],
    random_seed=state.generate_dataset_params["seed"],
    services=state.generate_dataset_params["services"],
    ports=state.generate_dataset_params["ports"],
)

with config_params_col_2:
    st.write("Example graph in dataset" if state.is_dataset else "Graph")
    fig = generator.show_network()
    st.pyplot(fig)

    with st.popover("View Legend"):
        st.markdown("**Node Types**")
        for node_type, attributes in NODE_TYPE_MAP.items():
            color = attributes["colour"]
            st.markdown(f'<span style="color:{color};">■</span> {node_type.name}', unsafe_allow_html=True)

# st.sidebar.divider()
# st.sidebar.write("For Debug:")
# st.sidebar.write(state.generate_dataset_params)

