import numpy as np
import threading
import random


class ParameterHandler:
    def __init__(self, clientJS, latent):
        self.data_dir = "./latent_param_dataset_16.npz"
        self.crossfade_threshold = 0.02
        self.buffer_duration_seconds = 0.05


        self.clientJS = clientJS
        # Track last time we sent a drawBox to compute elapsed duration for previous box
        self._last_drawbox_time = None

        # Visualization change speed tracking
        self.prev_draw_coords = None
        latent = np.asarray(latent)
        # Precompute min and range per dimension to normalize to [0,1]
        self._latent_min = np.min(latent, axis=0)
        self._latent_ptp_vec = np.ptp(latent, axis=0)
        self._latent_ptp_vec[self._latent_ptp_vec == 0] = 1.0  # avoid div by zero
        # For delta normalization, use the Euclidean length of the latent bounding box diagonal
        self._latent_range_norm = float(np.linalg.norm(self._latent_ptp_vec)) if np.any(self._latent_ptp_vec) else 1.0
        
        # Buffer for target_params with short delay
        self.target_params_buffer = []
        self.buffer_timer = None
        self.buffer_lock = threading.Lock()

    def getparameters_handler(self, address: str, *args):
        print(f"Received msg on address {address} with args {args}")
        # Convert args to integers (removes decimal points)
        target_params = np.array(args, dtype=int)
        print(f"Target params as integers: {target_params}")
        self._buffer_target_params(target_params)

    def _buffer_target_params(self, target_params):
        """Buffer target_params and set up short timer to process the last one"""
        with self.buffer_lock:
            # Add the new target_params to buffer
            self.target_params_buffer.append(target_params)

            # Cancel existing timer if it exists
            if self.buffer_timer is not None:
                self.buffer_timer.cancel()

            self.buffer_timer = threading.Timer(self.buffer_duration_seconds, self._process_buffered_params)
            self.buffer_timer.start()

    def _process_buffered_params(self):
        """Process the last buffered target_params and clear the buffer"""
        with self.buffer_lock:
            if self.target_params_buffer:
                # Get the last (most recent) target_params
                last_target_params = self.target_params_buffer[-1]
                # Clear the buffer
                self.target_params_buffer.clear()
                # Reset timer
                self.buffer_timer = None

                print(f"Processing buffered params after delay: {last_target_params}")
                self.find_closest_point(last_target_params)

    def find_closest_point(self, target_params):
        dataset = np.load(self.data_dir)

        # Convert both to integers for consistent comparison
        param_matrix_int = dataset['parameter_matrix'].astype(int)
        target_params_int = np.array(target_params, dtype=int)

        # Ensure target_params is 1D and has correct length
        if target_params_int.ndim > 1:
            target_params_int = target_params_int.flatten()

        # If incoming has fewer params, compare against first N
        if len(target_params_int) != param_matrix_int.shape[1]:
            if len(target_params_int) < param_matrix_int.shape[1]:
                param_matrix_int = param_matrix_int[:, :len(target_params_int)]

        # Find exact match using broadcasting
        matches = np.all(param_matrix_int == target_params_int[np.newaxis, :], axis=1)
        print(f"Number of exact matches found: {np.sum(matches)}")

        # Determine selected index (exact match preferred; otherwise closest by distance)
        if np.any(matches):
            selected_index = np.where(matches)[0][0]
            print(f"Exact match found at index {selected_index}")
        else:
            print("No exact match found even after integer conversion")
            # Fall back to closest match (use integer-converted target params)
            distances = np.linalg.norm(param_matrix_int - target_params_int, axis=1)
            selected_index = np.argmin(distances)
            print(f"Closest match at index {selected_index}")

        x, y, z = dataset['reduced_latent_matrix'][selected_index]
        # Normalize to [0,1] using precomputed dataset bounds so they render correctly in the UI
        x_n = float((x - self._latent_min[0]) / self._latent_ptp_vec[0])
        y_n = float((y - self._latent_min[1]) / self._latent_ptp_vec[1])
        z_n = float((z - self._latent_min[2]) / self._latent_ptp_vec[2])
        print(f"Latent coordinates (raw): x={x}, y={y}, z={z}")
        print(f"Latent coordinates (normalized): x={x_n}, y={y_n}, z={z_n}")
        try:
            # Decide visualization based on how big the change is vs. the previous draw coords
            if self.prev_draw_coords is not None:
                prev = np.array(self.prev_draw_coords, dtype=float)
                curr = np.array([x_n, y_n, z_n], dtype=float)
                delta = float(np.linalg.norm(curr - prev))
                # Normalize by latent space diagonal to get a scale-invariant measure
                normalized_delta = delta / self._latent_range_norm if self._latent_range_norm else delta
                print(f"Change magnitude (normalized): {normalized_delta:.4f} (threshold {self.crossfade_threshold})")
                if normalized_delta > self.crossfade_threshold:
                    # Fast change
                    self.clientJS.send_message("/drawCrossfade", "")
                else:
                    # Slow change
                    self.clientJS.send_message("/drawMeander", "")
            else:
                # First draw; treat as slow/meander by default
                self.clientJS.send_message("/drawMeander", "")

            # Compute elapsed seconds since last drawBox
            import time
            now = time.time()
            elapsed_prev_sec = None
            if self._last_drawbox_time is not None:
                elapsed_prev_sec = float(now - self._last_drawbox_time)
            self._last_drawbox_time = now

            # Draw the box at the normalized coordinates and include elapsed_prev_sec for previous box
            # If elapsed_prev_sec is None, send -1 to indicate unknown (first box)
            elapsed_arg = elapsed_prev_sec if elapsed_prev_sec is not None else -1.0
            print(f"prev seconds: {elapsed_arg} index is {selected_index}")
            # 5th argument should be the dataset index for color lookup in the frontend
            self.clientJS.send_message("/drawBox", [x, y, z, random.randint(3, 9), int(selected_index), elapsed_arg])
            print(f"Sent drawBox message to Node.js (normalized): x={x}, y={y}, z={z}, prev_elapsed={elapsed_arg}")
            # Update previous draw coordinates using normalized values
            self.prev_draw_coords = [x_n, y_n, z_n]
        except Exception as e:
            print(f"Error sending to Node.js: {e}")
