import torch
from torch.utils.data import Dataset
from torch_geometric.data.batch import Batch
from primaite.agents.aegis.modules.llm import LLM
from typing import List


class GLLMDataset(Dataset):
    def __init__(self, graphs, questions, gt_answers, llm: LLM):
        self.graphs = graphs
        self.questions = questions
        self.gt_answers = gt_answers
        self.llm = llm
        self.gt_hidden_states = self._get_target_hidden_states(gt_answers)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return {
            "graph": self.graphs[idx],
            "question": self.questions[idx],
            "gt_answer": self.gt_answers[idx],
            "gt_hidden_states": self.gt_hidden_states[idx],
        }

    def _get_target_hidden_states(self, gt_answers: List[str]):
        hidden_states = torch.empty(0)
        for gt_answer in gt_answers:
            inputs = self.llm.get_input_embeddings(prompts=[gt_answer], grad=False)
            hidden_states = torch.cat(
                [
                    hidden_states,
                    self.llm.get_output_embeddings(inputs_embeds=inputs["inputs_embeds"], grad=False).to("cpu"),
                ],
                dim=0,
            )
        return hidden_states


def collate_fn(batch):
    graph_data_list = [item["graph"] for item in batch]
    questions = [item["question"] for item in batch]

    gt_answers = [item["gt_answer"] for item in batch]
    graph_batch = Batch.from_data_list(graph_data_list)
    gt_hidden_states = torch.cat([item["gt_hidden_states"] for item in batch], dim=0).unsqueeze(1)
    return {
        "graphs": graph_batch,
        "questions": questions,
        "gt_answers": gt_answers,
        "gt_hidden_states": gt_hidden_states,
    }
