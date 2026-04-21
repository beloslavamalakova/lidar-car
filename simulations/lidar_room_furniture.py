import struct
import math
import os
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# =========================
# CONFIG
# =========================
ROOM_SIZE = 10.0
SCAN_FREQ = 6
SAMPLE_FREQ = 4000
DURATION_SEC = 240          # 4 minutes
POINTS_PER_PACKET = 12
ANGLE_STEP = 360.0 * SCAN_FREQ / SAMPLE_FREQ   # 0.54 deg
MAX_RANGE = 8.0
PACKET_SIZE = 47

# =========================
# GLOBAL DRAWING STATE
# =========================
objects = []   # each object is dict: {"type": "rect"/"circle", ...}
lidar_pos = None

mode = "rect"          # rect / circle / lidar
temp_point = None      # first click for current object

fig, ax = plt.subplots(figsize=(8, 8))


# =========================
# GEOMETRY HELPERS
# =========================
def ray_intersect_room(x0, y0, dx, dy, room_size):
    t_candidates = []

    if abs(dx) > 1e-12:
        t1 = (0 - x0) / dx
        y1 = y0 + t1 * dy
        if t1 > 0 and 0 <= y1 <= room_size:
            t_candidates.append(t1)

        t2 = (room_size - x0) / dx
        y2 = y0 + t2 * dy
        if t2 > 0 and 0 <= y2 <= room_size:
            t_candidates.append(t2)

    if abs(dy) > 1e-12:
        t3 = (0 - y0) / dy
        x3 = x0 + t3 * dx
        if t3 > 0 and 0 <= x3 <= room_size:
            t_candidates.append(t3)

        t4 = (room_size - y0) / dy
        x4 = x0 + t4 * dx
        if t4 > 0 and 0 <= x4 <= room_size:
            t_candidates.append(t4)

    return min(t_candidates) if t_candidates else None


def ray_intersect_rect(x0, y0, dx, dy, rect):
    rx = rect["x"]
    ry = rect["y"]
    rw = rect["w"]
    rh = rect["h"]

    t_candidates = []

    # vertical edges
    if abs(dx) > 1e-12:
        for x_edge in [rx, rx + rw]:
            t = (x_edge - x0) / dx
            if t > 0:
                y = y0 + t * dy
                if ry <= y <= ry + rh:
                    t_candidates.append(t)

    # horizontal edges
    if abs(dy) > 1e-12:
        for y_edge in [ry, ry + rh]:
            t = (y_edge - y0) / dy
            if t > 0:
                x = x0 + t * dx
                if rx <= x <= rx + rw:
                    t_candidates.append(t)

    return min(t_candidates) if t_candidates else None


def ray_intersect_circle(x0, y0, dx, dy, circle):
    cx = circle["cx"]
    cy = circle["cy"]
    r = circle["r"]

    # Solve |(x0,y0) + t(dx,dy) - (cx,cy)|^2 = r^2
    ox = x0 - cx
    oy = y0 - cy

    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r

    disc = b * b - 4 * a * c
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    valid = [t for t in [t1, t2] if t > 1e-9]
    return min(valid) if valid else None


def get_distance(angle_deg, sensor_x, sensor_y):
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    candidates = []

    # room walls
    t_room = ray_intersect_room(sensor_x, sensor_y, dx, dy, ROOM_SIZE)
    if t_room is not None:
        candidates.append(t_room)

    # furniture
    for obj in objects:
        if obj["type"] == "rect":
            t = ray_intersect_rect(sensor_x, sensor_y, dx, dy, obj)
        elif obj["type"] == "circle":
            t = ray_intersect_circle(sensor_x, sensor_y, dx, dy, obj)
        else:
            t = None

        if t is not None:
            candidates.append(t)

    if not candidates:
        return None, 0

    dist = min(candidates)

    if dist > MAX_RANGE:
        return None, 0

    # small Gaussian noise
    dist += random.gauss(0, 0.005)
    dist = max(0.0, min(dist, MAX_RANGE))

    # simple intensity model: closer objects slightly stronger
    intensity = max(30, min(255, int(255 - 20 * dist)))
    return dist, intensity


# =========================
# BINARY PACKETS
# =========================
def generate_packet(start_angle, points):
    packet = bytearray(
        struct.pack(
            '<BBHH',
            0x54,
            0x2C,
            int(360 * SCAN_FREQ),
            int(start_angle * 100)
        )
    )

    for dist, intensity in points:
        if dist is None:
            packet += struct.pack('<HB', 0, 0)
        else:
            packet += struct.pack('<HB', int(dist * 1000), intensity)

    end_angle = (start_angle + ANGLE_STEP * (POINTS_PER_PACKET - 1)) % 360
    packet += struct.pack('<H', int(end_angle * 100))
    packet += struct.pack('<B', 0x00)   # dummy checksum
    packet += b'\x00\x00'               # padding

    return packet


def run_simulation():
    if lidar_pos is None:
        print("Place the LiDAR first.")
        return

    os.makedirs("data", exist_ok=True)
    file_path = "data/lidar_furniture_sim.bin"

    sensor_x, sensor_y = lidar_pos
    total_packets = (SAMPLE_FREQ * DURATION_SEC) // POINTS_PER_PACKET
    current_angle = 0.0

    print(f"Starting simulation with LiDAR at ({sensor_x:.2f}, {sensor_y:.2f})")
    print(f"Objects: {len(objects)}")
    print("Generating 4-minute scan...")

    with open(file_path, "wb") as f:
        for _ in range(total_packets):
            pts = []
            start_angle_packet = current_angle

            for _ in range(POINTS_PER_PACKET):
                d, intensity = get_distance(current_angle, sensor_x, sensor_y)
                pts.append((d, intensity))
                current_angle = (current_angle + ANGLE_STEP) % 360

            f.write(generate_packet(start_angle_packet, pts))

    print(f"Done. Saved to: {file_path}")
    print(f"File size: {os.path.getsize(file_path)/1024/1024:.2f} MB")

    visualize_lidar(file_path)


# =========================
# VISUALIZATION OF RESULT
# =========================
def visualize_lidar(bin_file):
    x_coords, y_coords = [], []

    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} not found.")
        return

    with open(bin_file, 'rb') as f:
        data = f.read()

    i = 0
    while i <= len(data) - PACKET_SIZE:
        if data[i] == 0x54 and data[i+1] == 0x2C:
            try:
                start_angle = struct.unpack('<H', data[i+4:i+6])[0] / 100.0

                for j in range(POINTS_PER_PACKET):
                    offset = i + 6 + (j * 3)
                    dist_raw = struct.unpack('<H', data[offset:offset+2])[0]
                    intensity = data[offset+2]

                    if dist_raw == 0 or intensity == 0:
                        continue

                    dist_m = dist_raw / 1000.0
                    angle_deg = (start_angle + j * ANGLE_STEP) % 360
                    angle_rad = math.radians(angle_deg)

                    x_coords.append(dist_m * math.cos(angle_rad))
                    y_coords.append(dist_m * math.sin(angle_rad))

                i += PACKET_SIZE
            except Exception:
                i += 1
        else:
            i += 1

    plt.figure(figsize=(8, 8))
    plt.style.use('dark_background')
    plt.scatter(x_coords, y_coords, s=2, alpha=0.35)
    plt.scatter(0, 0, c='red', marker='x', s=100, label='LiDAR')
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.title("LiDAR Reconstruction with Furniture")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


# =========================
# DRAWING UI
# =========================
def redraw():
    ax.clear()
    ax.set_xlim(0, ROOM_SIZE)
    ax.set_ylim(0, ROOM_SIZE)
    ax.set_aspect('equal')
    ax.set_title(
        f"Mode: {mode} | r=rectangle, c=circle, l=lidar, u=undo, enter=run"
    )
    ax.grid(True, alpha=0.3)

    # room border
    room_patch = Rectangle((0, 0), ROOM_SIZE, ROOM_SIZE, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(room_patch)

    # furniture
    for obj in objects:
        if obj["type"] == "rect":
            patch = Rectangle((obj["x"], obj["y"]), obj["w"], obj["h"], alpha=0.5)
            ax.add_patch(patch)
        elif obj["type"] == "circle":
            patch = Circle((obj["cx"], obj["cy"]), obj["r"], alpha=0.5)
            ax.add_patch(patch)

    # LiDAR
    if lidar_pos is not None:
        ax.plot(lidar_pos[0], lidar_pos[1], 'rx', markersize=12, markeredgewidth=3)

    fig.canvas.draw_idle()


def on_click(event):
    global temp_point, lidar_pos

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    x, y = event.xdata, event.ydata

    if mode == "lidar":
        lidar_pos = (x, y)
        redraw()
        return

    if temp_point is None:
        temp_point = (x, y)
        return

    x1, y1 = temp_point
    x2, y2 = x, y

    if mode == "rect":
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        if rw > 0.05 and rh > 0.05:
            objects.append({
                "type": "rect",
                "x": rx,
                "y": ry,
                "w": rw,
                "h": rh
            })

    elif mode == "circle":
        r = math.hypot(x2 - x1, y2 - y1)
        if r > 0.05:
            objects.append({
                "type": "circle",
                "cx": x1,
                "cy": y1,
                "r": r
            })

    temp_point = None
    redraw()


def on_key(event):
    global mode, temp_point

    if event.key == 'r':
        mode = "rect"
        temp_point = None
    elif event.key == 'c':
        mode = "circle"
        temp_point = None
    elif event.key == 'l':
        mode = "lidar"
        temp_point = None
    elif event.key == 'u':
        if objects:
            objects.pop()
    elif event.key == 'enter':
        plt.close(fig)
        run_simulation()
        return

    redraw()


# =========================
# MAIN
# =========================
def main():
    print("Interactive LiDAR Room Builder")
    print("Instructions:")
    print(" - Press r for rectangle furniture")
    print(" - Press c for circular furniture")
    print(" - Press l to place LiDAR")
    print(" - Click twice to create object")
    print(" - Press u to undo last object")
    print(" - Press Enter to start 4-minute scan")

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
