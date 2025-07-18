"""
Script to convert multiple USC WidowX datasets (potentially with different tasks)
into a single LeRobot dataset v2.0 format.

Can accept either directories containing task data (inferring trajectories within)
or explicit paths to individual trajectory directories.

Example usage (by task directories):
uv run examples/usc_widowx/convert_usc_data_to_lerobot.py --raw-dirs /path/to/task1 /path/to/task2 --repo-id <org>/<combined-dataset-name>

Example usage (by specific trajectories):
uv run examples/usc_widowx/convert_usc_data_to_lerobot.py --traj-paths /path/to/task1/traj0 /path/to/task2/traj5 --repo-id <org>/<combined-dataset-name>

Example again for new widowx:
python examples/usc_widowx/convert_usc_data_to_lerobot.py \
    --raw-dirs /Volumes/Sandisk\ 1TB/test_widowx_data/2025-06-19_19-25-31/raw/reach_the_green_block/ \
    --repo-id "jesbu1/usc_widowx_6_19_lerobot" \
    --mode "video" \
    --push-to-hub \
    --dataset-config.use-videos=True
"""

import dataclasses
import re
from pathlib import Path
import shutil
from typing import Literal, Dict, List, Tuple, Optional
import warnings
import pickle

try:
    # for older lerobot versions before 2.0.0
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

    OLD_LEROBOT = True
except ImportError:
    # newer lerobot versions use HF_LEROBOT_HOME instead of LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME

    OLD_LEROBOT = False
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
from PIL import Image
import torchvision.transforms.functional as F


# TODO(user): If your .dat files are not simple numpy saves, replace this.
def load_data_compressed(filepath: Path) -> Dict[str, np.ndarray]:
    """Placeholder function to load compressed data."""
    try:
        # Assuming .dat files might be numpy archives or pickled objects
        return np.load(filepath, allow_pickle=True).item()  # Use .item() if it's a saved dictionary
    except Exception as e:
        warnings.warn(f"Failed to load {filepath} with numpy.load: {e}. Implement custom loading if needed.")
        raise


def load_pickle_data(filepath: Path) -> Dict[str, np.ndarray]:
    """Load data from a pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


# TODO(user): Adapt this if image loading is different.
def load_images(image_dir: Path) -> np.ndarray:
    """Load images from a directory."""
    image_files = list(image_dir.glob("*.png"))  # Assuming PNG format, adjust if needed
    if not image_files:
        image_files = list(image_dir.glob("*.jpg"))

    # sort the image files by number
    image_files = sorted(
        image_files,
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )

    if not image_files:
        raise FileNotFoundError(f"No image files (.png, .jpg) found in {image_dir}")

    imgs = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        imgs.append(np.array(img))
    return np.stack(imgs)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001  # Adjust based on data timestamp precision
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    # TODO(user): Define image shape expected by LeRobot
    image_height: int = 256
    image_width: int = 256


DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_trajectory_paths(raw_dirs: List[Path]) -> List[Tuple[Path, str]]:
    """Find all trajectory directories within the list of raw data directories
       and associate them with a task name derived from the parent directory."""
    all_traj_infos = []
    for raw_dir in raw_dirs:
        if not raw_dir.is_dir():
            warnings.warn(f"Provided raw directory path is not a directory, skipping: {raw_dir}")
            continue

        # Assuming the task name is the name of the directory containing traj folders
        task_name = raw_dir.name
        task_name_processed = task_name.replace("_", " ").capitalize() # Process for better readability
        print(f"Processing task: {task_name_processed}")

        # Find trajectory folders (e.g., traj0, traj1, ...) within this raw_dir
        traj_paths_in_dir = [p for p in raw_dir.iterdir() if p.is_dir() and re.match(r"traj\d+", p.name)]

        if not traj_paths_in_dir:
            warnings.warn(f"No trajectory subdirectories found in {raw_dir}")
            continue

        print(f"Found {len(traj_paths_in_dir)} potential trajectory directories in {raw_dir} for task '{task_name_processed}'.")
        for traj_path in traj_paths_in_dir:
            all_traj_infos.append((traj_path, task_name_processed))

    if not all_traj_infos:
        raise FileNotFoundError(f"No trajectory subdirectories found across all provided raw directories: {raw_dirs}")

    print(f"Found a total of {len(all_traj_infos)} trajectories across all directories.")
    return all_traj_infos


def create_empty_dataset(
    repo_id: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # TODO(user): Verify motor names and count for WidowX.
    state = [
        # Example names, replace with actual motor names used in your data
        "x",
        "y",
        "z",
        "x_angle",
        "y_angle",
        "z_angle",
        "gripper",
    ]
    cameras = [
        "images0",
        # "external",
        # "over_shoulder",
        # Add other camera names if present, e.g., "wrist"
    ]

    features = {
        # Corresponds to obs_dict['state'] in the TFDS script
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state),),
            "names": state,
        },
        # Corresponds to policy_out['actions']
        "action": {
            "dtype": "float32",
            "shape": (len(state),),
            "names": state,
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, dataset_config.image_height, dataset_config.image_width),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        print(f"Removing existing dataset directory: {LEROBOT_HOME / repo_id}")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # TODO(user): Set the correct robot_type for WidowX
    robot_type = "widowx"

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(
    traj_path: Path,
    cameras: List[str],
    dataset_config: DatasetConfig,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load state, action, and image data for a single trajectory."""
    obs_file = traj_path / "obs_dict.pkl"
    action_file = traj_path / "policy_out.pkl"

    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file}")
    if not action_file.exists():
        raise FileNotFoundError(f"Action file not found: {action_file}")
    obs_data = load_pickle_data(obs_file)
    action_data = load_pickle_data(action_file)

    assert isinstance(action_data, list)
    assert isinstance(obs_data, dict)

    # Ensure data is numpy before converting to tensor
    state_np = obs_data["state"]
    action_np = np.concatenate([step["actions"][None] for step in action_data], axis=0)
    if not isinstance(state_np, np.ndarray):
        raise TypeError(f"Expected 'state' in {obs_file} to be numpy array, got {type(state_np)}")
    if not isinstance(action_np, np.ndarray):
        raise TypeError(f"Expected 'actions' in {action_file} to be numpy array, got {type(action_np)}")

    state = torch.from_numpy(state_np)  # Key based on TFDS script
    action = torch.from_numpy(action_np)  # Key based on TFDS script

    # Check lengths
    num_frames = state.shape[0]
    if action.shape[0] != num_frames:
        warnings.warn(
            f"State ({num_frames}) and action ({action.shape[0]}) length mismatch in {traj_path}. Truncating to minimum."
        )
        min_len = min(num_frames, action.shape[0])
        state = state[:min_len]
        action = action[:min_len]
        num_frames = min_len

    imgs_per_cam = {}
    for camera in cameras:
        # Assuming images are in a subdirectory named after the camera.
        image_dir = traj_path / f"{camera}"
        imgs_np = load_images(image_dir)
        print(f"Loaded {imgs_np.shape[0]} images from directory {image_dir}")

        # Preprocess images: center crop and resize
        if imgs_np.size == 0:
            warnings.warn(f"Empty image array loaded from {image_dir}. Skipping camera {camera}.")
            continue

        n, h, w, c = imgs_np.shape
        if c != 3:
            warnings.warn(f"Expected 3 channels, got {c} in {image_dir}. Skipping camera {camera}.")
            continue

        # Convert to tensor (N, C, H, W), normalize to [0, 1]
        imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0

        # Center crop to square
        crop_size = min(h, w)
        cropped_tensor = F.center_crop(imgs_tensor, output_size=[crop_size, crop_size])

        # Resize to target dimensions
        resized_tensor = F.resize(
            cropped_tensor, size=[dataset_config.image_height, dataset_config.image_width], antialias=True
        )

        # Store the processed tensor (N, C, H_out, W_out)
        imgs_per_cam[camera] = resized_tensor


        # Verify image count
        if camera in imgs_per_cam and imgs_per_cam[camera].shape[0] != num_frames:
            # if it's 1 more than the number of frames, just remove the last frame
            if imgs_per_cam[camera].shape[0] == num_frames + 1:
                imgs_per_cam[camera] = imgs_per_cam[camera][:-1]
            else:
                warnings.warn(
                    f"Image count ({imgs_per_cam[camera].shape[0]}) for camera {camera} does not match state/action count ({num_frames}) in {traj_path}. Skipping camera."
                )
                del imgs_per_cam[camera]

    # Return only cameras that were successfully loaded and matched frame count
    valid_cameras = list(imgs_per_cam.keys())
    if not valid_cameras:
        # If no valid cameras remain (e.g., all had frame count mismatches), return None for images
        warnings.warn(f"No valid image data found for any camera in {traj_path}. Returning None for images.")
        return None, state, action # Return state/action in case they are still useful without images

    # Ensure all required cameras are present after validation
    missing_required = [cam for cam in cameras if cam not in valid_cameras]
    if missing_required:
         warnings.warn(f"Required cameras {missing_required} missing valid data in {traj_path}. Returning None for images.")
         return None, state, action


    return imgs_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    traj_infos: List[Tuple[Path, str]], # List of (trajectory_path, task_name)
    dataset_config: DatasetConfig,
    episodes: Optional[List[int]] = None,
) -> LeRobotDataset:
    """Populate the LeRobotDataset with data from trajectory files."""
    if episodes is None:
        # If specific episodes are requested, filter traj_infos
        # Note: This applies indices across *all* found trajectories, not per task/directory
        selected_traj_infos = [traj_infos[i] for i in range(len(traj_infos))]
    else:
        if any(i >= len(traj_infos) for i in episodes):
             raise IndexError(f"Episode index out of bounds. Requested indices {episodes}, but found {len(traj_infos)} total trajectories.")
        selected_traj_infos = [traj_infos[i] for i in episodes]


    # Get camera names from dataset features
    cameras = [key.split(".")[-1] for key in dataset.features if key.startswith("observation.images.")]

    num_added_episodes = 0
    # Iterate through the selected (trajectory path, task name) tuples
    for ep_idx, (traj_path, task) in enumerate(tqdm.tqdm(selected_traj_infos, desc="Processing trajectories")):
        print(f"\nProcessing trajectory: {traj_path.name} (Task: {task})")
        try:
            loaded_data = load_raw_episode_data(traj_path, cameras, dataset_config)
            # Check if image loading failed
            if loaded_data[0] is None:
                 warnings.warn(f"Skipping trajectory {traj_path.name} due to missing/invalid image data.")
                 continue
            imgs_per_cam, state, action = loaded_data

        except FileNotFoundError as e:
            warnings.warn(f"Skipping trajectory {traj_path.name}: {e}")
            continue
        except TypeError as e:
             warnings.warn(f"Skipping trajectory {traj_path.name} due to data type error: {e}")
             continue
        except Exception as e:
             warnings.warn(f"Skipping trajectory {traj_path.name} due to unexpected error: {e}")
             continue


        num_frames = state.shape[0]

        if num_frames == 0:
            warnings.warn(f"Skipping empty trajectory: {traj_path.name}")
            continue
        # Basic check (more robust checks happen in load_raw_episode_data)
        if not imgs_per_cam or cameras[0] not in imgs_per_cam or num_frames != len(imgs_per_cam[cameras[0]]):
             warnings.warn(f"Frame count mismatch or missing camera data for {traj_path.name}. State: {num_frames}, Action: {action.shape[0]}, Images: {len(imgs_per_cam.get(cameras[0], []))}. Skipping.")
             continue


        for i in range(num_frames):
            frame = {
                "observation.state": state[i].numpy().astype(np.float32),
                "action": action[i].numpy().astype(np.float32),
            }

            all_cams_present = True
            for camera in cameras:
                if camera not in imgs_per_cam:
                    warnings.warn(
                        f"Camera {camera} missing image data for frame {i} in {traj_path.name}. Skipping frame."
                    )
                    all_cams_present = False
                    break  # Skip this frame if any required camera is missing
                # Ensure image is in CHW format for LeRobotDataset
                img = imgs_per_cam[camera][i]  # This is now a CHW tensor

                # Assign the CHW tensor directly
                frame[f"observation.images.{camera}"] = img

            assert all_cams_present, f"Camera {camera} missing image data for frame {i} in {traj_path.name}. Skipping frame."

            if not OLD_LEROBOT:
                dataset.add_frame(frame, task=task)
            else:
                dataset.add_frame(frame)

        if OLD_LEROBOT:
            dataset.save_episode(task=task)
        else:
            dataset.save_episode()
        num_added_episodes += 1
        print(f"Saved episode {num_added_episodes} from {traj_path.name} with {num_frames} frames.")

    print(f"Finished processing. Added {num_added_episodes} episodes.")
    return dataset


# TODO(USER): The logic below assumes that the parent directory of the trajectory path is the task name.
# Adjust if your structure differs when providing explicit paths.
def process_explicit_traj_paths(traj_paths: List[Path]) -> List[Tuple[Path, str]]:
    """Validate explicit trajectory paths and infer task names from parent directories."""
    all_traj_infos = []
    for traj_path in traj_paths:
        if not traj_path.is_dir():
            warnings.warn(f"Provided trajectory path is not a directory, skipping: {traj_path}")
            continue

        # Basic check if it looks like a trajectory directory based on name
        if not re.match(r"traj\d+", traj_path.name):
             warnings.warn(f"Provided path {traj_path} doesn't match 'traj<number>' pattern. Including it anyway, but task name inference might be incorrect if not in a task parent directory.")

        # Infer task name from the parent directory
        task_name = traj_path.parent.name
        task_name_processed = task_name.replace("_", " ").capitalize() # Process for better readability

        print(f"Found trajectory: {traj_path} (Task: '{task_name_processed}')")
        all_traj_infos.append((traj_path, task_name_processed))

    if not all_traj_infos:
        raise FileNotFoundError(f"No valid trajectory directories found in the provided list: {traj_paths}")

    print(f"Processed {len(all_traj_infos)} explicitly provided trajectories.")
    return all_traj_infos


def port_usc_data(
    repo_id: str,
    *,
    # Group for input specification
    raw_dirs: Optional[List[Path]] = None, # Specify parent directories containing task/trajectory folders
    traj_paths: Optional[List[Path]] = None, # OR specify explicit paths to trajectory folders
    # Other arguments
    raw_repo_id: Optional[str] = None, # Optional: HF repo to download raw data from if local paths don't exist (primarily for --raw-dirs)
    episodes: Optional[List[int]] = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Main function to convert USC data from multiple sources and optionally push to Hub."""

    # --- Input Validation ---
    if (raw_dirs is None and traj_paths is None) or (raw_dirs is not None and traj_paths is not None):
         raise ValueError("Please provide either --raw-dirs (parent directories) or --traj-paths (specific trajectory paths), but not both.")

    if (LEROBOT_HOME / repo_id).exists():
        print(f"Dataset already exists locally at {LEROBOT_HOME / repo_id}. Removing.")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    traj_infos: List[Tuple[Path, str]] = []

    if raw_dirs is not None:
        print("Processing using --raw-dirs mode.")
        # Check if *any* of the raw_dirs exist. Downloading is complex with multiple dirs.
        # Let's require directories to exist locally for now.
        # TODO(user): Add support for downloading multiple raw_repo_ids if needed.
        existing_raw_dirs = [d for d in raw_dirs if d.exists()]
        if not existing_raw_dirs:
            raise FileNotFoundError(f"None of the specified raw directories exist: {raw_dirs}")

        # Warn if some directories were provided but don't exist
        missing_dirs = [d for d in raw_dirs if not d.exists()]
        if missing_dirs:
            warnings.warn(f"The following raw directories do not exist and will be skipped: {missing_dirs}")


        # Use only the existing directories
        if not existing_raw_dirs:
             print("No existing raw directories found after download check. Exiting.")
             return
        traj_infos = get_trajectory_paths(existing_raw_dirs)

    elif traj_paths is not None:
         print("Processing using --traj-paths mode.")
         # Downloading specific trajectory paths is not supported via raw_repo_id currently.
         if raw_repo_id:
              warnings.warn("--raw-repo-id is ignored when using --traj-paths. Ensure all specified trajectory paths exist locally.")

         existing_traj_paths = [p for p in traj_paths if p.exists()]
         missing_traj_paths = [p for p in traj_paths if not p.exists()]

         if missing_traj_paths:
              warnings.warn(f"The following trajectory paths do not exist and will be skipped: {missing_traj_paths}")

         if not existing_traj_paths:
              raise FileNotFoundError(f"None of the specified trajectory paths exist: {traj_paths}")

         traj_infos = process_explicit_traj_paths(existing_traj_paths)


    if not traj_infos:
        print(f"No valid trajectories found based on the provided input. Exiting.")
        return

    dataset = create_empty_dataset(
        repo_id,
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        traj_infos, # Pass the list of (path, task) tuples
        dataset_config=dataset_config,
        episodes=episodes, # Note: episode indices apply to the *combined* list of trajectories found/provided
    )

    if dataset.num_episodes > 0:
        print("Consolidating dataset...")
        if OLD_LEROBOT:
            dataset.consolidate()

        if push_to_hub:
            print("Pushing dataset to Hugging Face Hub...")
            dataset.push_to_hub(
                repo_id=repo_id,
                use_videos=True,
                private=False,
                push_videos=True,
                upload_large_folder=True,
                license="apache-2.0",
            )
            print(f"Successfully pushed {repo_id} to Hub.")
        else:
            print(f"Dataset saved locally at {LEROBOT_HOME / repo_id}. Skipping push to Hub.")
    else:
        print("No episodes were successfully added to the dataset.")


if __name__ == "__main__":
    tyro.cli(port_usc_data)
