import struct
import math
import matplotlib.pyplot as plt
import os

def visualize_lidar(bin_file):
    x_coords, y_coords = [], []

    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} not found.")
        return

    with open(bin_file, 'rb') as f:
        data = f.read()

    i = 0
    while i < len(data) - 45:
        # Look for the header byte 0x54
        if data[i] == 0x54:
            try:
                # Extract start angle
                start_angle = struct.unpack('<H', data[i+4:i+6])[0] / 100.0

                for j in range(12):
                    offset = i + 6 + (j * 3)
                    dist_raw = struct.unpack('<H', data[offset : offset+2])[0]
                    dist_m = dist_raw / 1000.0

                    # Skip noise/max range points to see the square clearly
                    if dist_m > 0.1 and dist_m < 7.5:
                        angle_rad = math.radians((start_angle + (j * 0.54)) % 360)
                        x_coords.append(dist_m * math.cos(angle_rad))
                        y_coords.append(dist_m * math.sin(angle_rad))

                i += 45 # Move to next possible packet
            except:
                i += 1
        else:
            i += 1

    plt.figure(figsize=(8, 8))
    plt.style.use('dark_background')
    plt.scatter(x_coords, y_coords, s=5, c='#00FF00', alpha=0.5)
    plt.scatter(0, 0, c='red', marker='x', s=100, label='Sensor')
    plt.axis('equal')
    plt.title("Fixed Square Room Reconstruction")
    plt.show()

if __name__ == '__main__':
    visualize_lidar('data/lidar_sim.bin')
