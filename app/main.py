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

welcome_pg = st.Page("welcome.py", title="Welcome", icon="👋")
eval_pg = st.Page("evaluation.py", title="Evaluation", icon="🦧")
training_pg = st.Page("training.py", title="Training", icon="🐒")
analyse_py = st.Page("analyse.py", title="Analyse", icon="👓")
generate_py = st.Page("generate.py", title="Generate Network", icon="🧠")

# i dont really want analyse_py in the navbar
main_pg = st.navigation([welcome_pg, eval_pg, training_pg, analyse_py, generate_py])

st.set_page_config(page_title="PrimAITE", page_icon="🦍", layout="wide")


main_pg.run()
