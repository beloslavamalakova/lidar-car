import struct
import math
import matplotlib.pyplot as plt
import os

POINTS_PER_PACKET = 12
ANGLE_STEP = 0.54
PACKET_SIZE = 47

def visualize_lidar(bin_file):
    x_coords, y_coords = [], []

    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} not found.")
        return

    with open(bin_file, 'rb') as f:
        data = f.read()

    i = 0
    while i <= len(data) - PACKET_SIZE:
        # Safer header check
        if data[i] == 0x54 and data[i + 1] == 0x2C:
            try:
                start_angle = struct.unpack('<H', data[i+4:i+6])[0] / 100.0

                for j in range(POINTS_PER_PACKET):
                    offset = i + 6 + (j * 3)
                    dist_raw = struct.unpack('<H', data[offset:offset+2])[0]
                    dist_m = dist_raw / 1000.0

                    if 0.1 < dist_m < 7.5:
                        angle_deg = (start_angle + j * ANGLE_STEP) % 360
                        angle_rad = math.radians(angle_deg)

                        x = dist_m * math.cos(angle_rad)
                        y = dist_m * math.sin(angle_rad)

                        x_coords.append(x)
                        y_coords.append(y)

                i += PACKET_SIZE
            except Exception:
                i += 1
        else:
            i += 1

    plt.figure(figsize=(8, 8))
    plt.style.use('dark_background')
    plt.scatter(x_coords, y_coords, s=3, alpha=0.5)
    plt.scatter(0, 0, marker='x', s=100, label='Sensor')
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.title("Fixed Square Room Reconstruction (Sensor Frame)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

if __name__ == '__main__':
    visualize_lidar('data/lidar_sim.bin')
