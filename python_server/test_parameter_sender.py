#!/usr/bin/env python3
"""
Test script that sends parameters from the dataset to the parameter handler
every 20 seconds, cycling through all entries in the dataset.
"""

import numpy as np
import time
from pythonosc import udp_client


def main():
    # Load the dataset
    data_dir = "./latent_param_dataset_16.npz"
    dataset = np.load(data_dir)
    
    parameter_matrix = dataset['parameter_matrix']
    reduced_latent_matrix = dataset['reduced_latent_matrix']
    
    print(f"Loaded dataset with {len(parameter_matrix)} entries")
    print(f"Parameter matrix shape: {parameter_matrix.shape}")
    print(f"Reduced latent matrix shape: {reduced_latent_matrix.shape}")
    print("-" * 80)
    
    # Setup OSC client to send to the Python server
    ip = "127.0.0.1"
    port = 6666  # Port where the Python server listens
    client = udp_client.SimpleUDPClient(ip, port)
    
    print(f"OSC Client configured to send to {ip}:{port}")
    print(f"Starting to send parameters every 20 seconds...")
    print("=" * 80)
    
    # Cycle through dataset entries
    index = 0
    try:
        while True:
            # Get parameters and coordinates for current index
            params = parameter_matrix[index]
            x, y, z = reduced_latent_matrix[index]
            
            # Take first 8 parameters
            params_to_send = params[:8].astype(int)
            
            # Print information
            print(f"\n[{time.strftime('%H:%M:%S')}] Sending entry #{index}")
            print(f"  Parameters (first 8): {params_to_send.tolist()}")
            print(f"  Expected coordinates: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            
            # Send to getparameters_handler via OSC
            client.send_message("/getparameters", params_to_send.tolist())
            
            print(f"  ✓ Sent to /getparameters")
            print("-" * 80)
            
            # Move to next index (cycle back to 0 when reaching the end)
            index = (index + 1) % len(parameter_matrix)
            
            # Wait 20 seconds
            time.sleep(20)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
        print(f"Last index sent: {index - 1 if index > 0 else len(parameter_matrix) - 1}")


if __name__ == "__main__":
    main()
