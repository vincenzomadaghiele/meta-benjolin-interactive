import numpy as np
from scipy.spatial import KDTree
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import time
import threading
import random
from parameter_handler import ParameterHandler
from benjolin_synth import BenjolinSynth
import os
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import json
import mido


class LatentSpace():
    def __init__(self, dataset, clientPd, clientJS, dimensionality=3, k=150):
        self.dimensionality = dimensionality
        self.clientPd = clientPd
        self.clientJS = clientJS
        self.latent = np.squeeze(dataset['reduced_latent_matrix'][:, :self.dimensionality])
        self.parameter = np.squeeze(dataset['parameter_matrix'])
        self.kd_tree = KDTree(self.latent)
        self.current_index = None
        self.current_latent_coordinate = None
        self.current_parameters = None
        self.neighbors = k
        self.current_path = None
        self.path_cache = {}
        self.is_playing_crossfade = False
        self.is_playing_meander = False
        # Initialize parameter handler delegation
        self.param_handler = ParameterHandler(
            clientJS=self.clientJS,
            latent=self.latent
        )
        # Initialize BenjolinSynth with random startup parameters
        N_params = 9  # 8 benjolin parameters + gain
        startup_synth_parameters = np.random.rand(N_params).tolist()
        self.synth = BenjolinSynth(startup_synth_parameters)
        
        # Initialize MIDI controller state (8 parameters + gain)
        self.midi_parameters = [0.5] * 9  # Default to middle values (0-1 range)
        self.midi_port = None
        self.midi_thread = None
        self.midi_listening = False

        # Start MIDI listener automatically
        self.start_midi_listener()

    def play_benjo(self):
        #self.clientPd.send_message("/stop", 1)
        params_message = '-'.join([str(int(param)) for param in self.current_parameters])
        self.clientPd.send_message("/params", params_message)
        # Play synth with current parameters (scaled 0-1)
        self.midi_parameters = self.current_parameters
        print("Playing box with params: ", self.current_parameters)
        self.play_midi_parameters()

    def stop_benjo(self):
        print("Stop is called")
        self.clientPd.send_message("/stop", 0)
        if hasattr(self, 'synth') and self.synth:
            self.synth.stop()
    
    def play_midi_parameters(self):
        '''Play synth with current MIDI controller parameters'''
        if hasattr(self, 'synth') and self.synth:
            print(f"Playing MIDI parameters: {self.midi_parameters}")
            self.synth.play(self.midi_parameters)
    
    def _midi_listener(self):
        '''Background thread that listens to MIDI controller input'''
        try:
            while self.midi_listening:
                for msg in self.midi_port.iter_pending():
                    if msg.type == 'control_change':
                        # Map first 8 CC controllers (CC 0-7) to parameters 0-7
                        # CC 8 maps to gain (parameter 8)
                        if msg.control < 9:
                            # Convert MIDI value (0-127) to 0-1 range
                            #normalized_value = msg.value / 127.0
                            self.midi_parameters[msg.control] = msg.value
                            print(f"MIDI CC {msg.control}: {msg.value}")#-> {normalized_value:.3f}")
                            # Play synth with updated parameters
                            self.play_midi_parameters()
                            # Call getparameters_handler with MIDI parameters (only first 8, excluding gain)
                            try:
                                params_to_send = self.midi_parameters[:8]
                                print(f"Sending {len(params_to_send)} parameters to handler: {params_to_send}")
                                self.param_handler.getparameters_handler("midiListener", *params_to_send)
                            except Exception as param_error:
                                print(f"Error in getparameters_handler: {param_error}")
                time.sleep(0.01)  # Small delay to prevent CPU overuse
        except Exception as e:
            print(f"MIDI listener error: {e}")
            import traceback
            traceback.print_exc()
            self.midi_listening = False
    
    def start_midi_listener(self, port_name=None):
        '''Start listening to MIDI controller input'''
        try:
            # List available MIDI ports
            available_ports = mido.get_input_names()
            print(f"Available MIDI ports: {available_ports}")
            
            if not available_ports:
                print("No MIDI ports found")
                return False
            
            # Use specified port or first available port
            if port_name and port_name in available_ports:
                selected_port = port_name
            else:
                selected_port = available_ports[0]
            
            print(f"Opening MIDI port: {selected_port}")
            self.midi_port = mido.open_input(selected_port)
            self.midi_listening = True
            
            # Start MIDI listener in background thread
            self.midi_thread = threading.Thread(target=self._midi_listener, daemon=True)
            self.midi_thread.start()
            print("MIDI listener started")
            return True
            
        except Exception as e:
            print(f"Failed to start MIDI listener: {e}")
            return False
    
    def stop_midi_listener(self):
        '''Stop listening to MIDI controller'''
        self.midi_listening = False
        if self.midi_thread:
            self.midi_thread.join(timeout=1.0)
        if self.midi_port:
            self.midi_port.close()
        print("MIDI listener stopped")

    def start_recording(self):
        self.clientPd.send_message("/startrecording", 0)
    def stop_recording(self):
        self.clientPd.send_message("/stoprecording", 0)

    def get_current_index(self):
        return self.current_index
    
    def set_current_point(self, index):
        self.current_index = index
        self.current_latent_coordinate = self.latent[self.current_index, :]
        self.current_parameters = self.parameter[self.current_index, :]
        self.play_benjo()

    def get_point_info(self, index):
        return self.latent[index, :], self.parameter[index, :]
    
    def get_index_given_latent(self, latent):
        distance, index = self.kd_tree.query(latent, k=1)
        return index

    def pre_uniformize(self):
        x_ind_sorted = np.argsort(self.latent[:, 0])
        y_ind_sorted = np.argsort(self.latent[:, 1])

        n = len(self.latent)

        new_x = np.zeros((n))
        new_x[x_ind_sorted] = np.arange(n)
        new_y = np.zeros((n))
        new_y[y_ind_sorted] = np.arange(n)

        if self.dimensionality == 3:
            z_ind_sorted = np.argsort(self.latent[:, 2])
            new_z = np.zeros((n))
            new_z[z_ind_sorted] = np.arange(n)
            return np.column_stack((new_x, new_y, new_z)) / n

        new_points = np.column_stack((new_x, new_y)) / n
        return new_points
    
    def find_next_point(self, a, b, path):
        """
        Finds a point adjacent to a that is in the direction of b, with the least distance in parameter space

        Args:
            a - index of the current point
            b - index of the point to move towards
        """
        a_latent, a_param = self.get_point_info(a)
        b_latent, b_param = self.get_point_info(b)
        param_distance_0 = np.linalg.norm(a_param - b_param)
        latent_distance_0 = np.linalg.norm(a_latent - b_latent)

        k = self.neighbors
        distances, indices = self.kd_tree.query(a_latent, k=k)
        indices = np.squeeze(indices)
    
        param_space_distances = np.zeros((k))
        latent_space_distances = np.zeros((k))
        cost_values = np.zeros((k))
        constant = 0.0

        for i in range(k):
            index = indices[i]
            if index == b:
                return b
            if index in path:
                param_space_distances[i] = np.nan
                latent_space_distances[i] = np.nan
            else:
                k_latent, k_param = self.get_point_info(indices[i])
                param_distance_to_a = np.linalg.norm(k_param - a_param)
                latent_distance_to_b = np.linalg.norm(b_latent - k_latent)
                if latent_distance_to_b > latent_distance_0:
                    latent_distance_to_b = np.nan
                param_space_distances[i] = param_distance_to_a
                latent_space_distances[i] = latent_distance_to_b
            cost_values[i] = param_space_distances[i] + latent_space_distances[i]
        
        # Check if all values are NaN (no valid next point found)
        if np.all(np.isnan(cost_values)):
            print(f"Warning: All neighbors visited or invalid, jumping to target {b}")
            return b
        
        argmin = np.nanargmin(cost_values)
        # if np.any(np.isnan(cost_values)): print(cost_values)
        index_of_best = indices[argmin]
        
        if index_of_best == a:
            # If best point is current point, jump to target
            print(f"Warning: Best point is current point, jumping to target {b}")
            return b
        
        return index_of_best

    def calculate_meander(self, idx1, idx2):
        reached_goal = False
        path = np.array([idx1])
        steps = 0
        while not reached_goal:
            steps += 1
            if steps > 1000:
                return path
            new_point = self.find_next_point(idx1, idx2, path)
            path = np.append(path, new_point)
            idx1 = new_point
            if new_point == idx2:
                reached_goal == True
                break
        return path

    def get_meander(self, x1, y1, z1, x2, y2, z2):
        idx1 = self.get_index_given_latent([x1, y1, z1])
        idx2 = self.get_index_given_latent([x2, y2, z2])
        key = str(idx1) + "-" + str(idx2)
        if key in self.path_cache:
            return self.path_cache[key]
        else:
            path_of_indices = self.calculate_meander(idx1, idx2)
            self.path_cache[key] = path_of_indices
            return path_of_indices
        
    def play_box_handler(self, address: str, *args):
        #print(f'received msg: {address}, playing box coords {args[0]:.3f}, {args[1]:.3f} and {args[2]:.3f} ')
        x, y, z= args[0], args[1], args[2]
        index = self.get_index_given_latent([x, y, z])
        self.set_current_point(index=index)
        self.is_playing_crossfade = False
        self.is_playing_meander = False

    def play_meander_handler(self, address: str, *args):
        #print(f'received msg: {address}, playing meander coords {args[0]:.3f}, {args[1]:.3f}, {args[2]:.3f} --> {args[3]:.3f}, {args[3]:.3f}, {args[5]:.3f} in {args[6]:.2f} s')
        x1, y1, z1, x2, y2, z2, t = args[0], args[1], args[2], args[3], args[4], args[5], int(args[6])
        path_of_indices = self.get_meander(x1, y1, z1, x2, y2, z2)
        length = path_of_indices.shape[0]
        time_per_point = t / length

        # Set a flag to indicate that the function is running
        self.is_playing_meander = True
        self.is_playing_crossfade = False
        # Create a new thread to play the meander in the background
        thread = threading.Thread(target=self._play_meander_in_background, args=(length, path_of_indices, time_per_point))
        thread.start()

    def _play_meander_in_background(self, length, path_of_indices, time_per_point):
        self.clientPd.send_message("/stop", 1)
        for i in range(length):
            if not self.is_playing_meander:
                return
            self.set_current_point(path_of_indices[i])
            # params = cloud.parameter[path_of_indices[i], :]
            # params_message = '-'.join([str(int(param)) for param in params])
            # clientPd.send_message("/params", params_message)
            time.sleep(time_per_point)

    def play_crossfade_handler(self, address: str, *args):
        #print(f'received msg: {address}, playing crossfade coords {args[0]:.3f}, {args[1]:.3f} {args[2]:.3f} --> {args[3]:.3f}, {args[4]:.3f}, {args[5]:.3f} in {args[6]:.2f} s')
        x1, y1, z1, x2, y2, z2, t = args[0], args[1], args[2], args[3], args[4], args[5], int(args[6])
        idx1 = self.get_index_given_latent([x1, y1, z1])
        idx2 = self.get_index_given_latent([x2, y2, z2])
        _, params1 = self.get_point_info(index=idx1)
        _, params2 = self.get_point_info(index=idx2)
        time_per_point = 0.1
        steps = 10 * t

        # Set a flag to indicate that the function is running
        self.is_playing_crossfade = True
        self.is_playing_meander = False
        # Create a new thread to play the crossfade in the background
        thread = threading.Thread(target=self._play_crossfade_in_background, args=(params1, params2, steps, time_per_point))
        thread.start()

# shortest in parameters
    def _play_crossfade_in_background(self, params1, params2, steps, time_per_point):
        for i in range(steps):
            if not self.is_playing_crossfade:
                return
            b = i / steps
            a = 1 - b
            params = params1 * a + params2 * b
            params_message = '-'.join([str(int(param)) for param in params])
            clientPd.send_message("/params", params_message)
            #self.midi_parameters = params
            #self.play_midi_parameters()
            time.sleep(time_per_point)

    def drawMeander_handler(self, address: str, *args):
        #print(f'received msg: {address}, sending draw meander coords {args[0]:.3f}, {args[1]:.3f} --> {args[2]:.3f}, {args[3]:.3f}')
        x1, y1, z1, x2, y2, z2 = args[0], args[1], args[2], args[3], args[4], args[5]
        path_of_indices = self.get_meander(x1, y1, z1, x2, y2, z2)
        path_of_latents = self.latent[path_of_indices, :]

        #-------------------------NOTA BENE:-------------------------------
        # changed the lines below, this sends the index of each point in the path like:
        # path_message = "idx0-idx1-idx2..."
        # uses up less network but you need to get the coordinates in JS given the indices
        path_message = '-'.join([str(int(index)) for index in path_of_indices])

        # this could also work but takes up more network where we send each point's 
        # xyz-coordinates like "x0-y0-z0-x1-y1-z1-x2-y2-z2..." etc
        path_message = ""
        for point_coords in path_of_latents:
            path_message += ' '.join([str(coord) for coord in point_coords])
            path_message += ' '
            #print(point_coords)
        # print(path_message)

        # the code below is the old one
        # path_of_latents = self.parameter[path_of_indices, :]
        # path_of_indices_message = '-'.join([str(int(param)) for param in path_of_indices])

        self.clientJS.send_message("/meanderPath", path_message)

    def stop_handler(self, address: str):
        # print(f'received msg: {address}')
        self.stop_benjo()
        self.is_playing_crossfade = False
        self.is_playing_meander = False

    def startrecording_handler(self, address: str):
        #(f'received msg: {address}')
        self.start_recording()
        self.is_playing_crossfade = False
        self.is_playing_meander = False

    def stoprecording_handler(self, address: str):
        #print(f'received msg: {address}')
        self.stop_recording()
        self.is_playing_crossfade = False
        self.is_playing_meander = False

    def getparameters_handler(self, address: str, *args):
        return self.param_handler.getparameters_handler(address, *args)

    def trainDatasetWithColors(self):
        """Delegate to DataTrainer for backward compatibility."""
        try:
            # Support both package and script execution contexts
            from data_trainer import DataTrainer as _DT
        except Exception:
            _DT = DataTrainer
        trainer = _DT()
        return trainer.train_dataset_with_colors()



def default_handler(address, *args):
        print(f"DEFAULT {address}: {args}")


if __name__ == "__main__":
    # ENTER HERE THE DIRECTORY OF THE NPZ DATASET
    data_dir = "./latent_param_dataset_16.npz"
    dataset = np.load(data_dir)
    dimensionality = 3

    ip = "127.0.0.1"  # localhost
    send_port_pd = 8000  # must match the port declared in Pure data
    send_port_js = 8001 # must match the listen port in JS
    listen_port = 6666 # must match the send post in JS
    clientPd = udp_client.SimpleUDPClient(ip, send_port_pd)  # sender to Pd
    clientJS = udp_client.SimpleUDPClient(ip, send_port_js)  # sender to JS
    dispatcher = Dispatcher()
    server = BlockingOSCUDPServer((ip, listen_port), dispatcher)  # listener

    cloud = LatentSpace(dataset=dataset, clientPd=clientPd, clientJS=clientJS,
                         dimensionality=dimensionality)


    # dispatcher.map("/print", print_handler)
    dispatcher.map("/play/box", handler=cloud.play_box_handler)
    dispatcher.map("/play/meander", handler=cloud.play_meander_handler)
    dispatcher.map("/play/crossfade", handler=cloud.play_crossfade_handler)
    dispatcher.map("/draw/meander", handler=cloud.drawMeander_handler)
    dispatcher.map("/stop", handler=cloud.stop_handler)
    dispatcher.map("/startrecording", handler=cloud.startrecording_handler)
    dispatcher.map("/stoprecording", handler=cloud.stoprecording_handler)
    dispatcher.set_default_handler(handler=cloud.getparameters_handler)
    #cloud.trainDatasetWithColors()
    print("Set up complete! Start playing the benjolin!")
    server.serve_forever()  # Blocks forever
