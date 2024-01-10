import fire
import copy
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from eval import EmbeddingModelForEval, OnlineEvaluator
from layers import apply_laser, apply_laser_single

def main(
    model: str = "BAAI/bge-small-en-v1.5",
    target_module_strs: list[str] = ["intermediate.dense", "output.dense"],
    sizes: list[int] = [1],
    mteb_tasks: list[dict] = [
        # {"mteb_task": "Banking77Classification", "metric": ["test", "accuracy"]},
        {"mteb_task": "STS12", "metric": ["test", "cos_sim", "spearman"]},
    ],
):
    model = EmbeddingModelForEval(model)
    model.to(torch.device("mps"))
    evaluator = OnlineEvaluator(mteb_tasks)
    linear_layers = []
    for name, module in model.model.named_modules():
        # print(name)
        if isinstance(module, nn.Linear) and any([target_module_str in name for target_module_str in target_module_strs]):
            linear_layers.append(name)

    initial_result = evaluator.run(model)
    all_results = []
    print(f"Initial result: {initial_result}")
    print(f"Testing LASER on {len(linear_layers)} linear layers...")
    for linear_layer in linear_layers:
        for s in sizes:
            copied = copy.deepcopy(model)
            error = apply_laser_single(copied.model, linear_layer, s)
            copied.to(torch.device("mps"))
            result = evaluator.run(copied)
            all_results.append({
                "layer": linear_layer,
                "size": s,
                "result": result,
                "error": error
            })
    
    json.dump(all_results, open("results.json", "w"), indent=4)

    print("LASER-ing the 18 layers with best results...")
    best_results = sorted(all_results, key=lambda x: -x["result"]["STS12"])[:18]
    for result in best_results:
        apply_laser_single(model.model, result["layer"], result["size"])

    final_result = evaluator.run(model)
    print(f"Final result: {final_result}")

if __name__ == "__main__":
    fire.Fire(main)