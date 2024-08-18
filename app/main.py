import logging
import sys
from pathlib import Path

import streamlit as st

lay_down_config_root = Path("../data/laydown-configs/")
training_config_root = Path("../agents/training_configs/")
session_config_root = Path("../agents/trained_agents/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

eval_pg = st.Page("evaluation.py", title="Evaluation", icon="🦧")
training_pg = st.Page("training.py", title="Training", icon="🐒")

main_pg = st.navigation([eval_pg, training_pg])

st.set_page_config(page_title="PrimAITE", page_icon="🦍", layout="wide")


main_pg.run()
