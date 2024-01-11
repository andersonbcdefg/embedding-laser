import fire
import copy
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from eval import EmbeddingModelForEval, OnlineEvaluator
from laser import generate_proposals, evaluate_proposal, apply_laser

def step(
    model: nn.Module,
    num_modules: int,
    num_proposals: int,
    target_patterns: list[str],
    exclude_patterns: list[str],
    bottleneck_features: int,
    evaluator: OnlineEvaluator,
    min_score: float
):
    """
    In one step, we do the following:
    1. Generate a set of proposals based on the current value of bottleneck features
    2. Evaluate all the proposals
    3. If the best proposal is better than the minimum score, we permanently apply it to the model.
    4. If not, we return false to indicate that we should stop or increase bottleneck features.
    """
    proposals = generate_proposals(
        model,
        num_modules=num_modules,
        bottleneck_features=bottleneck_features,
        num_proposals=num_proposals,
        target_patterns=target_patterns,
        exclude_patterns=exclude_patterns
    )
    results = []
    for prop in proposals:
        assert not any(["laser" in x for x in prop]), "laser shouldn't be in the proposal"
        print(f"Applying LASER to {prop}...")
        result = evaluate_proposal(
            model,
            target_modules=prop,
            bottleneck_features=1,
            evaluator=evaluator
        )
        results.append(result)

    best_result = max(results, key=lambda x: x["score"])

    if best_result["score"] > min_score:
        print(f"Applying LASER to {best_result['layers']}. Achieved score {best_result['score']}.")
        apply_laser(model, best_result["layers"], bottleneck_features)
        return best_result["score"], best_result["layers"]
    else:
        return False, None


def main(
    model: str = "BAAI/bge-small-en-v1.5",
    target_patterns: list[str] = ["dense", "query", "key", "value"],
    mteb_tasks: list[dict] = [
        # {"mteb_task": "Banking77Classification", "metric": ["test", "accuracy"]},
        {"mteb_task": "STS12", "metric": ["test", "cos_sim", "spearman"]},
    ],
):
    model = EmbeddingModelForEval(model)
    model.to(torch.device("mps"))
    evaluator = OnlineEvaluator(mteb_tasks)
    initial_score, _ = evaluator.run(model)
    print("Initial score:", initial_score)
    initial_params = sum([x.numel() for x in model.parameters()])
    bottleneck_features = 1
    min_score = initial_score - 0.03
    lasered_layers = []
    for i in range(20):
        print(f"Step {i+1}:")
        score, layers = step(
            model,
            num_modules=2,
            num_proposals=40,
            target_patterns=target_patterns,
            exclude_patterns=["laser"],
            bottleneck_features=1,
            evaluator=evaluator,
            min_score=min_score
        )
        if not score:
            bottleneck_features *= 2
            min_score -= 0.005
            if bottleneck_features > 64:
                print("Reached max bottleneck features. Stopping.")
                break
            else:
                print(f"Score below threshold. Increasing bottleneck to {bottleneck_features} and decreasing min_score by 0.005.")
        else:
            lasered_layers.append({
                "layers": layers,
                "bottleneck_size": bottleneck_features,
            })

    final_score, _ = evaluator.run(model)
    final_params = sum([x.numel() for x in model.parameters()])
    print("Initial score:", initial_score)
    print("Final score:", final_score)
    print("Initial params:", initial_params)
    print("Final params:", final_params, "reduced to", final_params / initial_params * 100, "% of original.")
    json.dump(
        {
            "final_score": final_score,
            "initial_score": initial_score,
            "final_params": final_params,
            "initial_params": initial_params,
            "lasered_layers": lasered_layers
        },
        open("laser_results.json", "w")
    )

if __name__ == "__main__":
    fire.Fire(main)