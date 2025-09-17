# Instructions for Training Pi-0 on BRIDGE with PEEK:
We detail basic instructions below, and the original Pi-0 README is below this section.
## Basic Install:

```bash
uv venv
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install tensorflow tensorflow_datasets shapely openai # openai is for the PEEK evaluation
uv pip install git+https://github.com/memmelma/vila_utils.git # TODO: modify the vila_utils
```

## Training Instructions for PEEK or Pi-0 on BRIDGE-v2

Follow the below instructions to train Pi-0 on BRIDGE-v2 with PEEK.
If you want to just serve the policy, skip to the next section.

### Download the PEEK dataset
```bash
uv run python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; dataset = LeRobotDataset('jesbu1/bridge_v2_lerobot_pathmask')"
```

### Training with PEEK
You can train with PEEK on 4 GPUs and batch size of 256 (should work for 4 48GB GPUs) by running the following:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_lora_bridge_1_cam_path_masked --exp-name=EXP_NAME --overwrite [--resume if you want to resume training]
```

You can train the original Pi-0 on BRIDGE-v2 with PEEK on 4 GPUs and batch size of 256 (should work for 4 48GB GPUs) by running the following:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_lora_bridge_1_cam --exp-name=EXP_NAME --overwrite [--resume if you want to resume training]
```

If you want to change the # of GPUs or batch size, modify the `fsdp_devices` and `batch_size` in the config file for `pi0_lora_bridge_1_cam_path_masked` or `pi0_lora_bridge_1_cam` at `src/openpi/training/config.py`.
You can also change the `num_workers` in `src/openpi/training/config.py` to change the # of workers for data loading.

## Hosting the Server for Evaluation
Once done training, you can evaluate the model by running the following command to initialize a policy server:
```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_lora_bridge_1_cam_path_masked --policy.dir=checkpoints/pi0_lora_bridge_1_cam_path_masked/29999/ 

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_lora_bridge_1_cam_path_masked --policy.dir=checkpoints/pi0_lora_bridge_1_cam_path_masked/29999/
```

In a separate terminal, run the following command to run the Libero evaluation script:
```bash
# Create virtual environment
conda deactivate
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install -e ../vila_utils # from https://github.com/memmelma/vila_utils
uv pip install wandb
uv pip install openai shapely # for pathmask
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py --args.task_suite_name=libero_10 --args.draw_path --args.draw_mask --args.vlm_server_ip="https://whippet-pet-singularly.ngrok.app" --args.vlm_query_frequency=20
python examples/libero/main.py --args.task_suite_name=libero_spatial --args.draw_path --args.draw_mask --args.vlm_server_ip="https://whippet-pet-singularly.ngrok.app" --args.vlm_query_frequency=20
python examples/libero/main.py --args.task_suite_name=libero_object --args.draw_path --args.draw_mask --args.vlm_server_ip="https://whippet-pet-singularly.ngrok.app" --args.vlm_query_frequency=20
python examples/libero/main.py --args.task_suite_name=libero_goal --args.draw_path --args.draw_mask --args.vlm_server_ip="https://whippet-pet-singularly.ngrok.app" --args.vlm_query_frequency=20

python examples/libero/main.py --args.task_suite_name=libero_spatial --args.draw_path --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio"
python examples/libero/main.py --args.task_suite_name=libero_object --args.draw_path --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio"
python examples/libero/main.py --args.task_suite_name=libero_goal --args.draw_path --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio"
python examples/libero/main.py --args.task_suite_name=libero_10 --args.draw_path --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio"


python examples/libero/main.py --args.task_suite_name=libero_spatial --args.draw_path --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8003
python examples/libero/main.py --args.task_suite_name=libero_object --args.draw_path --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8003
python examples/libero/main.py --args.task_suite_name=libero_goal --args.draw_path --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8003
python examples/libero/main.py --args.task_suite_name=libero_10 --args.draw_path --args.vlm_server_ip="http://0.0.0.0:8002" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8003

python examples/libero/main.py --args.task_suite_name=libero_spatial --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8004" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8005
python examples/libero/main.py --args.task_suite_name=libero_object --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8004" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8005
python examples/libero/main.py --args.task_suite_name=libero_goal --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8004" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8005
python examples/libero/main.py --args.task_suite_name=libero_10 --args.draw_mask --args.vlm_server_ip="http://0.0.0.0:8004" --args.vlm_query_frequency=20 --args.wandb_name_suffix="no_proprio" --args.port 8005
```

# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains two types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA)
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.

For both models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base diffusion [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `s3://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `s3://openpi-assets/checkpoints/pi0_fast_base` |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case  | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/), can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `s3://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/), faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well | `s3://openpi-assets/checkpoints/pi0_droid` |
| $\pi_0$-ALOHA-towel      | Inference | $\pi_0$ model fine-tuned on internal ALOHA data, can fold diverse towels 0-shot on [ALOHA](https://tonyzhaozh.github.io/aloha/) robot platforms                                                          | `s3://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference | $\pi_0$ model fine-tuned on internal ALOHA data, can unpack food from a tupperware container                                                                                                             | `s3://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference | $\pi_0$ model fine-tuned on [public ALOHA data](https://dit-policy.github.io/), can uncap a pen                                                                                                                                    | `s3://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |


By default, checkpoints are automatically downloaded from `s3://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for converting data, training, and running inference on various robots:
* [DROID](examples/droid/README.md)
* [ALOHA](examples/aloha_real/README.md)
* [USC WidowX](examples/usc_widowx/README.md)
* [Libero](examples/libero/README.md) (Data conversion and training only)

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_0$-FAST model on the [Libero dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting Libero data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw Libero dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for Libero below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the Libero environment to the model and vice versa. Will be used for both, training and inference.