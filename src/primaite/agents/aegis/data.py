import torch
from torch.utils.data import Dataset
from torch_geometric.data.batch import Batch
from primaite.agents.aegis.modules.llm import LLM
from typing import List


class GLLMDataset(Dataset):
    def __init__(self, graphs, questions, gt_answers, llm: LLM):
        self.graphs = graphs
        self.questions = questions
        self.gt_answers = [
            answer + llm.tokenizer.eos_token for answer in gt_answers
        ]  # mock add the eos token to the gt responses
        self.llm = llm
        self.gt_hidden_states = self._get_target_hidden_states(gt_answers)
        self.gt_logprobs, self.gt_tokens = self._get_target_logprobs_and_tokens(gt_answers)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return {
            "graph": self.graphs[idx],
            "question": self.questions[idx],
            "gt_answer": self.gt_answers[idx],
            "gt_hidden_states": self.gt_hidden_states[idx],
            "gt_logprobs": self.gt_logprobs[idx],
            "gt_tokens": self.gt_tokens[idx],
        }

    def _get_target_hidden_states(self, gt_answers: List[str]):
        inputs = self.llm.get_input_embeddings(prompts=gt_answers, grad=False)
        hidden_states = self.llm.get_output_embeddings(inputs_embeds=inputs["inputs_embeds"], grad=False).to("cpu")
        return hidden_states

    def _get_target_logprobs_and_tokens(self, gt_answers: List[str]):
        # Construct mock logprobs for gt repsonse
        gt_tokens = self.llm.tokenizer.batch_encode_plus(gt_answers, return_tensors="pt", padding=True)["input_ids"]
        batch_size = gt_tokens.shape[0]
        gt_probs = torch.zeros((batch_size, gt_tokens.shape[1], len(self.llm.tokenizer)), dtype=torch.float32)
        gt_probs.scatter_(2, gt_tokens.unsqueeze(-1), 1.0)
        return gt_probs, gt_tokens


def collate_fn(batch):
    graph_data_list = [item["graph"] for item in batch]
    questions = [item["question"] for item in batch]

    gt_answers = [item["gt_answer"] for item in batch]
    graph_batch = Batch.from_data_list(graph_data_list)
    gt_hidden_states = torch.cat([item["gt_hidden_states"] for item in batch], dim=0).unsqueeze(1)
    gt_logprobs = torch.stack([item["gt_logprobs"] for item in batch], dim=0)
    gt_tokens = torch.stack([item["gt_tokens"] for item in batch], dim=0)
    return {
        "graphs": graph_batch,
        "questions": questions,
        "gt_answers": gt_answers,
        "gt_hidden_states": gt_hidden_states,
        "gt_logprobs": gt_logprobs,
        "gt_tokens": gt_tokens,
    }
