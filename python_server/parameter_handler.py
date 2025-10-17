import numpy as np
import threading
import random


class ParameterHandler:
    def __init__(self, clientJS, latent):
        self.data_dir = "./latent_param_dataset_16.npz"
        self.change_threshold = 0.02
        self.buffer_duration_seconds = 0.1


        self.clientJS = clientJS
        # Track last time we sent a drawBox to compute elapsed duration for previous box
        self._last_drawbox_time = None

        # Visualization change speed tracking
        self.prev_draw_coords = None
        
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

    def _calculate_param_change_duration(self):
        """Calculate how long parameters were changing in the buffer.
        
        This algorithm detects when parameters have permanently stabilized by looking
        for a sustained period of identical values at the end of the buffer.
        It includes temporary pauses in the change duration.
        """
        buffer_len = len(self.target_params_buffer)
        if buffer_len <= 1:
            # Only one entry or empty, no change duration
            return 0
        
        # Find the stabilization point by looking backwards from the end
        # Count how many consecutive identical entries exist at the end
        stabilization_threshold = max(2, int(buffer_len * 0.3))  # At least 30% of buffer or 2 entries
        stable_count = 1
        
        for i in range(buffer_len - 1, 0, -1):
            if np.array_equal(self.target_params_buffer[i], self.target_params_buffer[i - 1]):
                stable_count += 1
            else:
                # Found a change, stop counting stable entries
                break
        
        # Determine if parameters have permanently stabilized
        if stable_count >= stabilization_threshold:
            # Parameters have stabilized - calculate duration up to stabilization point
            change_end_index = buffer_len - stable_count
            if change_end_index <= 0:
                # All entries are the same, no change
                return 0
            # Calculate proportion of buffer that had changes
            proportion = change_end_index / (buffer_len - 1)
            duration_ms = int(proportion * self.buffer_duration_seconds * 1000)
        else:
            # Parameters are still changing or haven't stabilized long enough
            # Use full buffer duration
            duration_ms = int(self.buffer_duration_seconds * 1000)
        
        print(f"Buffer analysis: {buffer_len} entries, {stable_count} stable at end, threshold: {stabilization_threshold}")
        duration_calculated = max(duration_ms, 0)  # Ensure non-negative
        return duration_calculated if duration_calculated > 0 else 100 

    def find_closest_point(self, target_params):
        dataset = np.load(self.data_dir)

        # Convert both to integers for consistent comparison
        param_matrix_int = dataset['parameter_matrix'].astype(int)
        target_params_int = np.array(target_params, dtype=int)

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
        print(f"Latent coordinates: x={x}, y={y}, z={z}")
        try:
            # Compute elapsed seconds since last drawBox
            import time
            now = time.time()
            elapsed_prev_sec = None
            if self._last_drawbox_time is not None:
                elapsed_prev_sec = float(now - self._last_drawbox_time)
            self._last_drawbox_time = now

            # Calculate parameter change duration
            change_duration_ms = self._calculate_param_change_duration()

            # Decide visualization based on how big the change is vs. the previous draw coords
            if self.prev_draw_coords is not None:
                prev = np.array(self.prev_draw_coords, dtype=float)
                curr = np.array([x, y, z], dtype=float)
                delta = float(np.linalg.norm(curr - prev))
                print(f"Change magnitude: {delta:.4f} (threshold {self.change_threshold})")
                if delta > self.change_threshold:
                    # Fast change - send crossfade with parameter change duration
                    self.clientJS.send_message("/drawCrossfade", change_duration_ms)
                    print(f"Sent /drawCrossfade with duration: {change_duration_ms}ms")
                else:
                    # Slow change - send meander with parameter change duration
                    self.clientJS.send_message("/drawMeander", change_duration_ms)
                    print(f"Sent /drawMeander with duration: {change_duration_ms}ms")
            else:
                self.clientJS.send_message("/drawMeander", change_duration_ms)
                print(f"Sent /drawMeander (first draw) with duration: {change_duration_ms}ms")

            # Draw the box at the coordinates and include elapsed_prev_sec for previous box
            # If elapsed_prev_sec is None, send -1 to indicate unknown (first box)
            elapsed_arg = elapsed_prev_sec if elapsed_prev_sec is not None else -1.0
            print(f"prev seconds: {elapsed_arg} index is {selected_index}")
            # 5th argument should be the dataset index for color lookup in the frontend
            self.clientJS.send_message("/drawBox", [x, y, z, random.randint(3, 9), int(selected_index), elapsed_arg])
            print(f"Sent drawBox message to Node.js: x={x}, y={y}, z={z}, prev_elapsed={elapsed_arg}")
            # Update previous draw coordinates using raw values
            self.prev_draw_coords = [x, y, z]
        except Exception as e:
            print(f"Error sending to Node.js: {e}")
