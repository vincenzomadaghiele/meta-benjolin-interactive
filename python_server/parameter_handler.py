import numpy as np
import threading
import random
import os
import sys
import time
import json

class ParameterHandler:
    def __init__(self, clientJS, latent, synth, encoding_mode=False):
        self.data_dir = "./latent_param_dataset_16.npz"
        self.change_threshold = 0.02
        self.midi_buffer_duration_seconds = 0.1
        self.encoding_mode = encoding_mode

        self.clientJS = clientJS
        self.latent_space = latent  # Store reference to latent space object
        self.synth = synth  # Store synth reference for playback
        # Track last time we sent a drawBox to compute elapsed duration for previous box
        self._last_drawbox_time = None

        # Visualization change speed tracking
        self.prev_draw_coords = None
        
        # Buffer for target_params with short delay
        self.target_params_buffer = []
        self.buffer_timer = None
        self.buffer_lock = threading.Lock()
        self.dataset_lock = threading.Lock()  # Lock for dataset file access
        
        # Initialize training components if training_mode is enabled
        self.encoder = None
        if self.encoding_mode:
            print("Training mode ENABLED")
            self._initialize_encoding_components()

    def _initialize_encoding_components(self):
        """Initialize BenjolinEncoder and BenjolinSynth for training mode."""
        try:
            # Add benjolin_encoder directory to path
            training_dir = os.path.join(os.path.dirname(__file__), 'benjolin_encoder')
            if training_dir not in sys.path:
                sys.path.insert(0, training_dir)
            
            from benjolin_encoder import BenjolinEncoder
            
            # Initialize trainer with model from benjolin_encoder directory
            model_path = os.path.join(training_dir, 'model')
            self.encoder = BenjolinEncoder(
                model_path=model_path,
                latent_dim=16,
                input_dim=36  # Must match the saved model's input dimension
            )
            print("BenjolinEncoder initialized successfully")
            print("Encoding mode ENABLED: Will use VAE encoding for unknown parameters")
            
        except Exception as e:
            print(f"Error initializing encoding components: {e}")
            print("Encoding mode will be DISABLED")
            self.encoding_mode = False
            import traceback
            traceback.print_exc()

    def getparameters_handler(self, source: str, *args):
        print(f"Received msg from source {source} with args {args}")
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

            self.buffer_timer = threading.Timer(self.midi_buffer_duration_seconds, self._process_buffered_params)
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
                self.find_and_send_coordinates(last_target_params)

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
            duration_ms = int(proportion * self.midi_buffer_duration_seconds * 1000)
        else:
            # Parameters are still changing or haven't stabilized long enough
            # Use full buffer duration
            duration_ms = int(self.midi_buffer_duration_seconds * 1000)
        
        print(f"Buffer analysis: {buffer_len} entries, {stable_count} stable at end, threshold: {stabilization_threshold}")
        duration_calculated = max(duration_ms, 0)  # Ensure non-negative
        return duration_calculated if duration_calculated > 0 else 100 

    def find_and_send_coordinates(self, target_params):
        # Load dataset with lock, then release it
        with self.dataset_lock:
            dataset = np.load(self.data_dir)
            # Convert both to integers for consistent comparison
            param_matrix_int = dataset['parameter_matrix'].astype(int)
            reduced_latent_matrix = dataset['reduced_latent_matrix'].copy()
        
        # All processing happens outside the lock
        target_params_int = np.array(target_params, dtype=int)

        # If incoming has fewer params, compare against first N
        if len(target_params_int) != param_matrix_int.shape[1]:
            if len(target_params_int) < param_matrix_int.shape[1]:
                param_matrix_int = param_matrix_int[:, :len(target_params_int)]

        # Find exact match using broadcasting
        matches = np.all(param_matrix_int == target_params_int[np.newaxis, :], axis=1)
        print(f"Number of exact matches found: {np.sum(matches)}")

        # Normalize parameters from 0-127 range to 0-1 range (BenjolinPatch expects normalized values)
        normalized_params = target_params_int / 127.0
        # Add gain parameter (1.5 for 150% volume - louder but without distortion)
        synth_params = np.append(normalized_params, 1).tolist()

        # Determine selected index (exact match preferred; otherwise closest by distance)
        if np.any(matches):
            selected_index = np.where(matches)[0][0]
            print(f"Exact match found at index {selected_index}")
            x, y, z = reduced_latent_matrix[selected_index]
        else:
            print("No exact match found even after integer conversion")
            
            # If training mode is enabled, generate coordinates from actual sound
            if self.encoding_mode and self.encoder is not None and self.synth is not None:
                print("Encoding mode: Generating coordinates from synthesized audio")
                try:
                    # Render audio with these parameters
                    audio_buffer = self.synth.render_audio_as_buffer(synth_params, duration_seconds=1.0)
                    
                    # Get 3D coordinates from the trainer
                    x, y, z = self.encoder.get_new_coordinates(audio_buffer)
                    print(f"Generated coordinates from audio: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                    
                    # Add new point to dataset and visualization (safe - lock is released)
                    selected_index = self._add_new_point_to_dataset(x, y, z, target_params_int)
                    
                except Exception as e:
                    print(f"Error in training mode coordinate generation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to closest match
                    print("Falling back to closest match in dataset")
                    distances = np.linalg.norm(param_matrix_int - target_params_int, axis=1)
                    selected_index = np.argmin(distances)
                    x, y, z = reduced_latent_matrix[selected_index]
                    print(f"Closest match at index {selected_index}")
            else:
                # Training mode disabled or not available - use closest match
                distances = np.linalg.norm(param_matrix_int - target_params_int, axis=1)
                selected_index = np.argmin(distances)
                # Manhattan distance alternative:
                #distances = np.sum(np.abs(param_matrix_int - target_params_int), axis=1)
                #selected_index = np.argmin(distances)
                x, y, z = reduced_latent_matrix[selected_index]
                print(f"Closest match at index {selected_index}")
        
        print(f"Latent coordinates: x={x}, y={y}, z={z}")
        if selected_index is not None and selected_index >= 0:
            print(f"Parameters of closest point: index {selected_index}")
        else:
            print(f"Generated point (not in dataset) with target parameters: {target_params_int}")
        try:
            # Compute elapsed seconds since last drawBox 
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
            self.synth.play(synth_params)
            self.clientJS.send_message("/drawBox", [x, y, z, random.randint(3, 9), int(selected_index), elapsed_arg])
            print(f"Sent drawBox message to Node.js: x={x}, y={y}, z={z}, prev_elapsed={elapsed_arg}")
            # Update previous draw coordinates using raw values
            self.prev_draw_coords = [x, y, z]
        except Exception as e:
            print(f"Error sending to Node.js: {e}")
    
    def _add_new_point_to_dataset(self, x, y, z, parameters):
        """Add new generated point to dataset files and send to frontend for visualization.
        
        Args:
            x, y, z: 3D coordinates
            parameters: 8-element parameter array
        """
        try:
            # Load existing dataset
            dataset = np.load(self.data_dir)
            
            # Extract existing matrices
            reduced_latent = dataset['reduced_latent_matrix']
            param_matrix = dataset['parameter_matrix']
            
            # Create new point arrays
            new_coord = np.array([[x, y, z]])
            new_param = np.array([parameters])
            
            # Append to existing data
            updated_reduced_latent = np.vstack([reduced_latent, new_coord])
            updated_param_matrix = np.vstack([param_matrix, new_param])
            
            # Get the index of the newly added point (last index)
            new_point_index = len(updated_reduced_latent) - 1
            
            # Save updated dataset
            np.savez(self.data_dir,
                    reduced_latent_matrix=updated_reduced_latent,
                    parameter_matrix=updated_param_matrix,
                    latent_matrix=dataset.get('latent_matrix', np.array([])),
                    sigma_matrix=dataset.get('sigma_matrix', np.array([])))
            
            print(f"Added new point to dataset at index {new_point_index}: coords=({x:.3f}, {y:.3f}, {z:.3f}), params={parameters}")
            
            # Send new point to frontend for dynamic addition
            self._send_new_point_to_frontend(x, y, z, parameters)
            
            # NOTE: Don't update dataset3D_withcolors.js during runtime as it causes dev server to reload the page
            # Instead, regenerate it from the .npz file when restarting the server or run a separate script
            # self._update_dataset3d_withcolors_js(x, y, z, r=1.0, g=0.0, b=0.0)  # Red color for new points
            
            # Reload dataset in latent space to update KD-tree
            if hasattr(self, 'latent_space') and self.latent_space:
                self.latent_space.reload_dataset()
            
            return new_point_index
            
        except Exception as e:
            print(f"Error adding point to dataset: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_dataset3d_withcolors_js(self, x, y, z, r, g, b):
        """Update the dataset3D_withcolors.js file with new coordinate and color."""
        try:
            js_file_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dataset3D_withcolors.js')
            
            # Read existing file
            with open(js_file_path, 'r') as f:
                content = f.read()
            
            # Find the x, y, z, r, g, b arrays and append new values
            import re
            
            # Add to x array - find last number before closing bracket
            x_pattern = r'("x":\s*\[[^\]]+)(\])'
            content = re.sub(x_pattern, f'\\1,\n        {x}\\2', content, count=1)
            
            # Add to y array
            y_pattern = r'("y":\s*\[[^\]]+)(\])'
            content = re.sub(y_pattern, f'\\1,\n        {y}\\2', content, count=1)
            
            # Add to z array
            z_pattern = r'("z":\s*\[[^\]]+)(\])'
            content = re.sub(z_pattern, f'\\1,\n        {z}\\2', content, count=1)
            
            # Add to r array
            r_pattern = r'("r":\s*\[[^\]]+)(\])'
            content = re.sub(r_pattern, f'\\1,\n        {r}\\2', content, count=1)
            
            # Add to g array
            g_pattern = r'("g":\s*\[[^\]]+)(\])'
            content = re.sub(g_pattern, f'\\1,\n        {g}\\2', content, count=1)
            
            # Add to b array
            b_pattern = r'("b":\s*\[[^\]]+)(\])'
            content = re.sub(b_pattern, f'\\1,\n        {b}\\2', content, count=1)
            
            # Write back
            with open(js_file_path, 'w') as f:
                f.write(content)
            
            print(f"Updated dataset3D_withcolors.js with new point (red color)")
            
        except Exception as e:
            print(f"Error updating dataset3D_withcolors.js: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_new_point_to_frontend(self, x, y, z, parameters):
        """Send new point to frontend for real-time visualization update."""
        try:
            message_data = {
                'type': 'new_point',
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'parameters': parameters.tolist()
            }
            self.clientJS.send_message("/newPoint", json.dumps(message_data))
            print(f"Sent new point to frontend for visualization")
        except Exception as e:
            print(f"Error sending new point to frontend: {e}")
