import asyncio
import datetime
import base64
import logging
import traceback
import os
import numpy as np
from PIL import Image
import time
import cv2
import concurrent.futures
from functools import partial
from collections import deque
from typing import Dict, Any


from openai import OpenAI, APIConnectionError
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames

from peek_vlm import add_answer_to_img, send_request

POLICY_INPUT_RESOLUTION = 224
PEEK_VLM_NAME = "vila_3b_path_mask_fast"

def get_path_mask_from_vlm(
    image: np.ndarray,
    task_instr: str,
    draw_path=True,
    draw_mask=True,
    vlm_server_ip: str = None,
    current_vlm_pred: str = None,
):
    # used for VLM inference during eval
    assert draw_path or draw_mask
    assert current_vlm_pred is not None or vlm_server_ip is not None, "Either current_vlm_pred or vlm_server_ip must be provided"
    prompt_type = "path_mask"
    pil_image = Image.fromarray(image)
    if not current_vlm_pred:
        # query the VLM otherwise use the provided path and mask
        answer_pred = send_request(pil_image, task_instr, prompt_type=prompt_type, server_ip=vlm_server_ip, model_name=PEEK_VLM_NAME)
    else:
        answer_pred = current_vlm_pred

    H, W, _ = pil_image.shape
    line_size = int(min(H, W) * 0.01)
    mask_pixels = int(min(H, W) * 0.08)

    path_mask_image, _, _ = add_answer_to_img(
        pil_image, answer_pred, prompt_type, line_size=line_size, add_mask=True, mask_pixels=mask_pixels
    )
    return path_mask_image, answer_pred
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol with VLM integration and temporal ensembling.
    
    Provides temporal ensembling of action predictions similar to the HTTP server for improved
    prediction stability and reduced noise in action outputs.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        obs_remap_key: str | None = None,
        vlm_img_key: str | None = None,
        vlm_server_ip: str | None = None,
        vlm_query_frequency: int = 5,
        vlm_draw_path: bool = True,
        vlm_draw_mask: bool = True,
        action_chunk_history_size: int = 10,
        ensemble_window_size: int = 5,
        temporal_weight_decay: float = 0.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        self._obs_remap_key = obs_remap_key
        # VLM integration parameters
        self._vlm_img_key = vlm_img_key
        self._vlm_server_ip = vlm_server_ip
        self._vlm_query_frequency = int(vlm_query_frequency)
        self._vlm_draw_path = bool(vlm_draw_path)
        self._vlm_draw_mask = bool(vlm_draw_mask)
        self._vlm_current_pred  = None
        self._vlm_step = 0
        
        # Temporal ensembling parameters
        self._action_chunk_history_size = action_chunk_history_size
        self._ensemble_window_size = ensemble_window_size
        self._temporal_weight_decay = temporal_weight_decay
        
        # Rolling buffer for action chunks and observations
        self._action_chunk_history = deque(maxlen=action_chunk_history_size)
        self._observation_history = deque(maxlen=action_chunk_history_size)
        
        # VLM save directory setup
        self._vlm_save_dir = None
        if self._vlm_img_key is not None:
            self._vlm_save_dir = os.path.join(os.getcwd(), "vlm_tmp")
            os.makedirs(self._vlm_save_dir, exist_ok=True)
            logging.info(f"VLM images will be saved to: {self._vlm_save_dir}")
            print(f"🖼️ VLM images will be saved to: {self._vlm_save_dir}")
        
        # Initialize thread pool executor for background image saving
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,  # Limit to 2 workers to avoid overwhelming the system
            thread_name_prefix="VLMImageSaver"
        )
        
        logging.info(f"Initialized VLM websocket policy server with action chunk history size: {action_chunk_history_size}, ensemble window: {ensemble_window_size}")

    def __del__(self):
        """Cleanup method to properly shut down the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)

    def cleanup(self):
        """Explicit cleanup method to shut down the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logging.info("Thread pool executor shut down successfully")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def _save_vlm_images(self, obs, original_img, img, step):
        """Save both original and processed VLM images to separate subfolders in a background thread."""
        if self._vlm_save_dir is not None:
            # Submit the image saving task to the thread pool executor
            future = self._executor.submit(self._save_vlm_images_sync, obs, original_img, img, step)
            # Add a callback to log any errors that occur in the thread
            future.add_done_callback(self._log_save_result)

    def _save_vlm_images_sync(self, obs, original_img, img, step):
        """Synchronous version of image saving that runs in a separate thread."""
        try:
            # Save original image
            original_dir = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), 'original')
            os.makedirs(original_dir, exist_ok=True)
            original_save_name = f"original/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            original_save_path = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), original_save_name)
            Image.fromarray(original_img).save(original_save_path)
            logging.info(f"Saved original image to {original_save_path}")
            print(f"🖼️ Saved original image to {original_save_path}")
            
            # Save processed image
            processed_dir = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            processed_save_name = f"processed/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            processed_save_path = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), processed_save_name)
            Image.fromarray(img).save(processed_save_path)
            logging.info(f"Saved processed VLM image to {processed_save_path}")
            print(f"🖼️ Saved processed VLM image to {processed_save_path}")
        except Exception as save_err:
            logging.warning(f"Failed to save VLM images: {save_err}")
            print(f"❌ Failed to save VLM images: {save_err}")
            raise  # Re-raise to be caught by the callback

    def _log_save_result(self, future):
        """Callback to log the result of the image saving operation."""
        try:
            future.result()  # This will raise any exception that occurred
        except Exception as e:
            logging.error(f"Error in background image saving thread: {e}")
            print(f"❌ Error in background image saving thread: {e}")

    def _extract_action_chunk(self, action: Dict[str, Any]) -> np.ndarray:
        """Extract action chunk from policy response."""
        if "actions" in action:
            return np.array(action["actions"])
        elif "action" in action:
            return np.array(action["action"])
        else:
            # If no clear action chunk, use the entire action dict
            return np.array(list(action.values()))

    def _update_history(self, observation: Dict[str, Any], action_chunk: np.ndarray):
        """Update action chunk and observation history."""
        self._action_chunk_history.append(action_chunk.copy())
        self._observation_history.append(observation.copy())
        logging.debug(f"Updated history. Current size: {len(self._action_chunk_history)}")

    def _temporal_ensemble(self, current_action: Dict[str, Any], current_action_chunk: np.ndarray) -> Dict[str, Any]:
        """Perform temporal ensembling of action predictions."""
        if len(self._action_chunk_history) < self._ensemble_window_size:
            # Not enough history for ensembling, return current action
            return current_action
        
        # Get recent action chunks for ensemble
        recent_chunks = list(self._action_chunk_history)[-self._ensemble_window_size:]
        
        # Apply temporal weighting with decay
        weights = np.array([self._temporal_weight_decay ** i for i in range(len(recent_chunks))])
        weights = weights / weights.sum()  # Normalize weights
        
        # Weighted ensemble of action chunks
        ensemble_chunk = np.zeros_like(current_action_chunk)
        for i, chunk in enumerate(recent_chunks):
            if chunk.shape == current_action_chunk.shape:
                ensemble_chunk += weights[i] * chunk
            else:
                # Handle shape mismatches by using current chunk
                ensemble_chunk += weights[i] * current_action_chunk
        
        # Create ensemble action response
        ensemble_action = current_action.copy()
        if "actions" in ensemble_action:
            ensemble_action["actions"] = ensemble_chunk
        elif "action" in ensemble_action:
            ensemble_action["action"] = ensemble_chunk
        else:
            # Update all numeric values with ensemble
            for key, value in ensemble_action.items():
                if isinstance(value, (int, float, np.number)):
                    ensemble_action[key] = float(ensemble_chunk[0] if len(ensemble_chunk) > 0 else value)
        
        logging.info(f"Applied temporal ensemble with {len(recent_chunks)} chunks, weights: {weights}")
        return ensemble_action

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about temporal ensembling state."""
        return {
            "action_chunk_history_size": len(self._action_chunk_history),
            "observation_history_size": len(self._observation_history),
            "max_history_size": self._action_chunk_history_size,
            "ensemble_window_size": self._ensemble_window_size,
            "temporal_weight_decay": self._temporal_weight_decay,
            "recent_action_chunks": list(self._action_chunk_history)[-5:] if self._action_chunk_history else []
        }

    def reset_ensemble_history(self) -> Dict[str, Any]:
        """Reset temporal ensembling history programmatically."""
        self._action_chunk_history.clear()
        self._observation_history.clear()
        logging.info("Temporal ensembling history has been reset programmatically")
        return {
            "action_chunk_history_size": len(self._action_chunk_history),
            "observation_history_size": len(self._observation_history)
        }

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                # Handle reset
                if obs.get("reset", False):
                    logging.info(f"Resetting policy and VLM step")
                    self._policy.reset()
                    self._vlm_step = 0
                    # Also reset temporal ensembling history
                    self._action_chunk_history.clear()
                    self._observation_history.clear()
                    logging.info("Temporal ensembling history has been reset")
                
                # VLM image processing
                if self._vlm_img_key is not None and self._vlm_img_key in obs:
                    
                    try:
                        original_img = obs[self._vlm_img_key]
                        
                        if self._vlm_draw_path or self._vlm_draw_mask:
                            if self._vlm_step % self._vlm_query_frequency == 0:
                                try:
                                    img, self.current_vlm_pred = get_path_mask_from_vlm(
                                        image=original_img,
                                        task_instr=obs.get("prompt", ""),
                                        draw_path=self._vlm_draw_path,
                                        draw_mask=self._vlm_draw_mask,
                                        vlm_server_ip=self._vlm_server_ip,
                                    )
                                    success = True
                                except Exception as e:
                                    logging.warning(f"VLM overlay error on query: {e}")
                                    self._vlm_current_pred = None
                                    success = False
                                if success:
                                    # Save both the original and overlaid images for this fresh query
                                    self._save_vlm_images(obs, original_img, img, self._vlm_step)
                                else:
                                    img = original_img
                            elif self._vlm_current_pred is not None:
                                try:
                                    img, _ = get_path_mask_from_vlm(
                                        image=original_img,
                                        task_instr=obs.get("prompt", ""),
                                        draw_path=self._vlm_draw_path,
                                        draw_mask=self._vlm_draw_mask,
                                        vlm_server_ip=None,
                                        current_vlm_pred=self._vlm_current_pred,
                                    )
                                    success = True
                                except Exception as e:
                                    logging.warning(f"VLM overlay error on reuse: {e}")
                                    self._vlm_current_pred = None
                                    success = False
                                if success:
                                    # Save both the original and overlaid images for this fresh query
                                    self._save_vlm_images(obs, original_img, img, self._vlm_step)
                                else:
                                    img = original_img
                        # Update the image in the observation
                        obs[self._vlm_img_key] = img
                        # downsample
                        obs[self._vlm_img_key] = cv2.resize(obs[self._vlm_img_key], (POLICY_INPUT_RESOLUTION, POLICY_INPUT_RESOLUTION))
                        
                    finally:
                        self._vlm_step += 1

                # rename keys in observation
                if self._obs_remap_key is not None:
                    obs[self._obs_remap_key] = obs[self._vlm_img_key]
                    del obs[self._vlm_img_key]

                action = self._policy.infer(obs)
                
                # Extract action chunk for history
                action_chunk = self._extract_action_chunk(action)

                if self._temporal_weight_decay != 0: 
                    # Update history
                    self._update_history(obs, action_chunk)
                    
                    # Perform temporal ensembling if we have enough history
                    ensemble_action = self._temporal_ensemble(action, action_chunk)
                else:
                    ensemble_action = action
                
                await websocket.send(packer.pack(ensemble_action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
