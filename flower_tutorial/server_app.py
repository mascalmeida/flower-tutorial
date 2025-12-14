"""flower-tutorial: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad

from flower_tutorial.task import Net

# Import central_evaluate function for server-side evaluation
from flower_tutorial.task import central_evaluate
# Import custom strategy
from flower_tutorial.custom_strategy import CustomFedAdagrad

from datetime import datetime
from pathlib import Path

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    #strategy = FedAvg(fraction_train=fraction_train)
    # Initialize FedAdagrad strategy
    #strategy = FedAdagrad(fraction_train=fraction_train)
    # Initialize Custom FedAdagrad strategy
    strategy = CustomFedAdagrad(
        fraction_train=fraction_train,
        fraction_evaluate=0.05,  # Evaluate on 50 clients (each round)
        min_train_nodes=20,  # Optional config
        min_evaluate_nodes=40,  # Optional config
        min_available_nodes=1000,  # Optional config
    )

    # Get the current date and time
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Set the path where results and model checkpoints will be saved
    strategy.set_save_path(save_path)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
