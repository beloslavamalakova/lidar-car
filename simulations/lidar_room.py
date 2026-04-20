import struct
import math
import os
import random

"""
SENSOR SPECIFICATION (LD20):
- Scanning Frequency: 6 Hz
- Sampling Frequency: 4000 pts/sec
- Range: 8.0m Max
"""

# Configuration
ROOM_SIZE = 10.0           # 10m x 10m
SENSOR_X, SENSOR_Y = 5.0, 5.0  # Kept as requested
SCAN_FREQ = 6
SAMPLE_FREQ = 4000
DURATION_SEC = 240         # 4 Minutes
POINTS_PER_PACKET = 12
TOTAL_PACKETS = (SAMPLE_FREQ * DURATION_SEC) // POINTS_PER_PACKET

def get_distance_to_wall(angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    t_candidates = []

    # Check X walls (using ROOM_SIZE and SENSOR variables)
    if cos_a > 1e-6: # Right wall
        t_candidates.append((ROOM_SIZE - SENSOR_X) / cos_a)
    elif cos_a < -1e-6: # Left wall
        t_candidates.append((0.0 - SENSOR_X) / cos_a)

    # Check Y walls (using ROOM_SIZE and SENSOR variables)
    if sin_a > 1e-6: # Top wall
        t_candidates.append((ROOM_SIZE - SENSOR_Y) / sin_a)
    elif sin_a < -1e-6: # Bottom wall
        t_candidates.append((0.0 - SENSOR_Y) / sin_a)

    valid_t = [t for t in t_candidates if t > 0]
    if not valid_t:
        return 8.0

    dist = min(valid_t)

    # --- ADDING REALISM: SENSOR NOISE ---
    # Adds +/- 5mm of random jitter so walls aren't "perfect"
    noise = random.gauss(0, 0.005)
    dist += noise

    # Clip to max sensor range (8m)
    return min(dist, 8.0)

def generate_packet(start_angle, points):
    # Header(1), VerLen(1), Speed(2), StartAngle(2) = 6 bytes
    packet = bytearray(struct.pack('<BBHH', 0x54, 0x2C, 360*SCAN_FREQ, int(start_angle * 100)))

    # 12 points * 3 bytes (Dist_H + Intensity_B) = 36 bytes
    for dist, intensity in points:
        packet += struct.pack('<HB', int(dist * 1000), intensity)

    # End Angle(2), Checksum(1), Padding(2) = 5 bytes
    # Total = 6 + 36 + 5 = 47 bytes
    end_angle = (start_angle + (0.54 * POINTS_PER_PACKET)) % 360
    packet += struct.pack('<H', int(end_angle * 100))
    packet += struct.pack('<B', 0x00)
    packet += b'\x00\x00'

    return packet

def main():
    if not os.path.exists("data"): os.makedirs("data")
    file_path = "data/lidar_sim.bin"
    current_angle = 0.0

    print(f"Simulating 10x10 room with sensor at ({SENSOR_X}, {SENSOR_Y})...")
    with open(file_path, "wb") as f:
        for _ in range(TOTAL_PACKETS):
            pts = []
            start_angle_packet = current_angle
            for _ in range(POINTS_PER_PACKET):
                d = get_distance_to_wall(current_angle)
                pts.append((d, 200))
                current_angle = (current_angle + 0.54) % 360
            f.write(generate_packet(start_angle_packet, pts))
    print(f"Done. File size: {os.path.getsize(file_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
