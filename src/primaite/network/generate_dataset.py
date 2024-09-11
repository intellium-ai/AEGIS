# external imports
from pathlib import Path
import os
from dotenv import load_dotenv
from torch_geometric.data import Batch
import torch
import random

# internal imports
from primaite.network.generator import NetworkGenerator
from primaite.agents.aegis.modules.openai import OpenAIClient

# constants
NUMBER_OF_NODES = (10, 50)
RANDOM_SEED = 165116
TRAINING_CONFIG_PATH = Path("../../../agents") / "training_configs" / "do_nothing.yaml"
SERVICE_NAMES = ["HTTP", "SSH", "FTP"]
PORTS_LIST = ["80", "22", "21"]
DATASET_SAVE_PATH = Path("../../../data") / "notebook_generated" / "overnight" / "data.pt"
DATASET_SIZE = 10000

load_dotenv()
open_ai_key = os.getenv('OPEN_AI_KEY')
openai = OpenAIClient(api_key=open_ai_key)

# generate datasets
outputs = []
fails = []
for i in range(DATASET_SIZE):

    try:
        laydown_save_path = Path("../../../data") / "notebook_generated" / "overnight" / f"{i}.yaml"
        generator = NetworkGenerator(graph_size=random.randint(NUMBER_OF_NODES[0], NUMBER_OF_NODES[1]), training_config_path=TRAINING_CONFIG_PATH, random_seed=1729, services=SERVICE_NAMES, ports=PORTS_LIST, pretrained_llm=openai)
        output = generator.run_end_to_end(laydown_save_path=laydown_save_path)
        outputs.append(output)
    except Exception as e:
        try:
            fails.append(str(e))
        except:
            pass

    print(f"\nPROGRESS -------> {i + 1} of {DATASET_SIZE} datasets generated\n")

    if i % 10:
        try:
            batch = Batch.from_data_list(outputs)
            torch.save(batch, DATASET_SAVE_PATH)
        except:
            pass

# save entire dataset as torch file
try:
    batch = Batch.from_data_list(outputs)
    torch.save(batch, DATASET_SAVE_PATH)
except:
    pass