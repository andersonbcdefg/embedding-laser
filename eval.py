import warnings
import tempfile
from typing import Union
from dataclasses import dataclass
import torch
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Task:
    mteb_task: str
    metric: list[str]  # list of keys to extract the metric


# example: Task(mteb_task="Banking77Classification", metric=["test", "accuracy"])
# example: Task(mteb_task="STS12", metric=["test", "cos_sim", "spearman"])

class OnlineEvaluator:
    def __init__(self, tasks: Union[list[Task], list[dict]]):
        super().__init__()
        if isinstance(tasks[0], dict):
            tasks = [Task(**task) for task in tasks]
        self.tasks = tasks

    def run(self, model):
        results = {}
        for task in self.tasks:
            evaluation = MTEB(tasks=[task.mteb_task], task_langs=["en"])
            eval_splits = ["dev"] if task.mteb_task == "MSMARCO" else ["test"]
            with tempfile.TemporaryDirectory() as tmpdirname:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = evaluation.run(
                        model, output_folder=tmpdirname, eval_splits=eval_splits
                    )[task.mteb_task]
            for key in task.metric:
                result = result[key]
            results[task.mteb_task] = result
        del model
        torch.cuda.empty_cache()
        return results

def pool(
    hiddens,
    pooled_output,
    mask,
    pooling_strategy: Literal["mean", "cls", "cls-mean-avg", "cls-mean-cat"],
    normalize: bool = True,
):
    # hiddens: B, L, D; mask: B, L
    if pooling_strategy == "mean":
        pooled = torch.sum(hiddens * mask.unsqueeze(-1), dim=1) / torch.sum(
            mask, dim=-1, keepdim=True
        )
    elif pooling_strategy == "cls":
        if pooled_output is None:
            # use first token w/ no pooling linear layer
            pooled = hiddens[:, 0, :]
        else:
            pooled = pooled_output
    elif pooling_strategy == "cls-mean-avg":
        if pooled_output is None:
            # use first token w/ no pooling linear layer
            cls_emb = hiddens[:, 0, :]
        else:
            cls_emb = pooled_output
        mean_emb = torch.mean(hiddens * mask.unsqueeze(-1), dim=1) / torch.sum(
            mask, dim=-1, keepdim=True
        )
        pooled = (cls_emb + mean_emb) / 2
    elif pooling_strategy == "cls-mean-cat": # WARNING: produces embeddings of size 2*D
        if pooled_output is None:
            # use first token w/ no pooling linear layer
            cls_emb = hiddens[:, 0, :]
        else:
            cls_emb = pooled_output
        mean_emb = torch.mean(hiddens * mask.unsqueeze(-1), dim=1) / torch.sum(
            mask, dim=-1, keepdim=True
        )
        pooled = torch.cat([cls_emb, mean_emb], dim=-1)
    else:
        raise ValueError(f"Pooling strategy {pooling_strategy} not supported")

    if normalize:
        pooled = F.normalize(pooled, dim=-1)

    return pooled

class EmbeddingModelForEval(nn.Module):
    """
    Wrapper for a model that returns embeddings. This is used for text embedding evals.
    NOT for training (use EmbeddingModelForFinetuning for that).
    """

    def __init__(
        self,
        path: str,
        pooling_strategy: Literal["mean", "cls", "cls-mean-avg", "cls-mean-cat"] = "mean",
        normalize: bool = True,
        max_length: int = 512,
        model_kwargs: Optional[dict] = {},
        tokenizer_kwargs: Optional[dict] = {}
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.normalize = normalize
        self.model.eval()
        
    @torch.no_grad()
    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        result = []
        for _, p in self.model.named_parameters():
            device = p.device
            break
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            tokenized = self.tokenizer(
                batch, max_length=self.max_length, padding="longest", truncation=True
            )
            input_ids = torch.Tensor(tokenized["input_ids"]).long().to(device)
            attention_mask = (
                torch.Tensor(tokenized["attention_mask"]).bool().to(device)
            )
            out: BaseModelOutputWithPoolingAndCrossAttentions = \
                self.model(input_ids, attention_mask, return_dict=True, **kwargs)
            hiddens, pooled_output = out.last_hidden_state, out.pooler_output
            embs = pool(
                hiddens, pooled_output, attention_mask, self.pooling_strategy, normalize=self.normalize
            )
            result.extend(embs.detach().cpu().to(torch.float32).numpy())

        return result
    
def test_eval():
    model = EmbeddingModelForEval("bert-base-uncased")
    evaluator = OnlineEvaluator([Task(mteb_task="Banking77Classification", metric=["test", "accuracy"])])
    results = evaluator.run(model)
    print(results)

if __name__ == "__main__":
    test_eval()