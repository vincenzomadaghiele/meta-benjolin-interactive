from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")

if __name__ == "__main__":

    # ENTER HERE THE DIRECTORY OF THE NPZ DATASET
    data_dir = "./latent_param_dataset_16.npz"
    dataset = np.load(data_dir)
    dimensionality = 3
    print(dataset['reduced_latent_matrix'].shape)
    print(dataset['parameter_matrix'].shape)
    xyz_matrix = dataset['reduced_latent_matrix']
    synthParameters_matrix = dataset['reduced_latent_matrix']

    ip = "127.0.0.1"  # localhost
    send_port_js = 8001 # must match the listen port in JS
    listen_port = 6666 # must match the send post in JS
    dispatcher = Dispatcher()
    server = BlockingOSCUDPServer((ip, listen_port), dispatcher)  # listener
    dispatcher.set_default_handler(default_handler)
    server.serve_forever()  # Blocks forever
