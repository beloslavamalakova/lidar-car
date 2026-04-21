import struct
import math
import os
import random
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# =========================
# CONFIG
# =========================
ROOM_SIZE = 10.0

SCAN_FREQ = 6
SAMPLE_FREQ = 4000
DURATION_SEC = 240
POINTS_PER_PACKET = 12
ANGLE_STEP = 360.0 * SCAN_FREQ / SAMPLE_FREQ   # 0.54 deg
MAX_RANGE = 8.0
PACKET_SIZE = 47

DT = 1.0 / SCAN_FREQ            # movement update once per full scan
SAFE_DISTANCE = 0.7             # more conservative than before
FRONT_CONE_DEG = 20             # wider than before
GOAL_TOLERANCE = 0.15
CAR_RADIUS = 0.12
EXTRA_MARGIN = 0.05             # extra safety
ANGLE_SEARCH_STEP = 5
MAX_SEARCH_ANGLE = 180

# =========================
# GLOBAL DRAWING STATE
# =========================
objects = []
lidar_pos = None
goal_pos = None

mode = "rect"   # rect / circle / lidar / goal
temp_point = None

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
    rx, ry, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
    t_candidates = []

    if abs(dx) > 1e-12:
        for x_edge in [rx, rx + rw]:
            t = (x_edge - x0) / dx
            if t > 0:
                y = y0 + t * dy
                if ry <= y <= ry + rh:
                    t_candidates.append(t)

    if abs(dy) > 1e-12:
        for y_edge in [ry, ry + rh]:
            t = (y_edge - y0) / dy
            if t > 0:
                x = x0 + t * dx
                if rx <= x <= rx + rw:
                    t_candidates.append(t)

    return min(t_candidates) if t_candidates else None


def ray_intersect_circle(x0, y0, dx, dy, circle):
    cx, cy, r = circle["cx"], circle["cy"], circle["r"]

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


def point_in_rect(x, y, rect, margin=0.0):
    return (
        rect["x"] - margin <= x <= rect["x"] + rect["w"] + margin and
        rect["y"] - margin <= y <= rect["y"] + rect["h"] + margin
    )


def point_in_circle(x, y, circle, margin=0.0):
    return math.hypot(x - circle["cx"], y - circle["cy"]) <= circle["r"] + margin


def collides(x, y):
    if x < CAR_RADIUS or x > ROOM_SIZE - CAR_RADIUS:
        return True
    if y < CAR_RADIUS or y > ROOM_SIZE - CAR_RADIUS:
        return True

    for obj in objects:
        if obj["type"] == "rect" and point_in_rect(x, y, obj, CAR_RADIUS):
            return True
        if obj["type"] == "circle" and point_in_circle(x, y, obj, CAR_RADIUS):
            return True

    return False


# =========================
# LIDAR DISTANCE MODEL
# =========================
def get_distance(angle_deg, sensor_x, sensor_y):
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    candidates = []

    t_room = ray_intersect_room(sensor_x, sensor_y, dx, dy, ROOM_SIZE)
    if t_room is not None:
        candidates.append(t_room)

    for obj in objects:
        if obj["type"] == "rect":
            t = ray_intersect_rect(sensor_x, sensor_y, dx, dy, obj)
        else:
            t = ray_intersect_circle(sensor_x, sensor_y, dx, dy, obj)

        if t is not None:
            candidates.append(t)

    if not candidates:
        return None, 0

    dist = min(candidates)

    if dist > MAX_RANGE:
        return None, 0

    # small measurement noise
    dist += random.gauss(0, 0.005)
    dist = max(0.0, min(dist, MAX_RANGE))

    intensity = max(30, min(255, int(255 - 20 * dist)))
    return dist, intensity


# =========================
# BINARY PACKET GENERATION
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


# =========================
# NAVIGATION HELPERS
# =========================
def heading_to_goal(x, y, gx, gy):
    return math.degrees(math.atan2(gy - y, gx - x)) % 360


def scan_sector_min_distance(sensor_x, sensor_y, center_deg, half_width_deg=10):
    min_d = float("inf")
    a = center_deg - half_width_deg

    while a <= center_deg + half_width_deg:
        d, _ = get_distance(a % 360, sensor_x, sensor_y)
        if d is not None:
            min_d = min(min_d, d)
        a += ANGLE_STEP

    return min_d if min_d != float("inf") else None


def build_candidate_offsets():
    offsets = [0]
    for k in range(1, int(MAX_SEARCH_ANGLE / ANGLE_SEARCH_STEP) + 1):
        offsets.append(k * ANGLE_SEARCH_STEP)
        offsets.append(-k * ANGLE_SEARCH_STEP)
    return offsets


CANDIDATE_OFFSETS = build_candidate_offsets()


def choose_heading(x, y, goal_x, goal_y, step_dist):
    desired = heading_to_goal(x, y, goal_x, goal_y)
    effective_clearance = SAFE_DISTANCE + CAR_RADIUS + EXTRA_MARGIN

    for off in CANDIDATE_OFFSETS:
        cand = (desired + off) % 360

        # 1. cone-based lidar clearance check
        d = scan_sector_min_distance(x, y, cand, FRONT_CONE_DEG)
        if d is not None and d <= effective_clearance:
            continue

        # 2. next-position collision check
        cand_rad = math.radians(cand)
        nx = x + step_dist * math.cos(cand_rad)
        ny = y + step_dist * math.sin(cand_rad)

        if collides(nx, ny):
            continue

        return cand, (off != 0)

    # fallback: no good direction found
    return None, True


# =========================
# MOVING SIMULATION
# =========================
def run_moving_simulation():
    if lidar_pos is None:
        print("Place the start/LiDAR point first.")
        return
    if goal_pos is None:
        print("Place the goal point first.")
        return
    if collides(lidar_pos[0], lidar_pos[1]):
        print("Start point is inside or too close to an obstacle.")
        return
    if collides(goal_pos[0], goal_pos[1]):
        print("Goal point is inside or too close to an obstacle.")
        return

    os.makedirs("data", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bin_path = f"data/lidar_moving_sim_{timestamp}.bin"
    csv_path = f"data/lidar_moving_path_{timestamp}.csv"
    png_path = f"data/lidar_moving_result_{timestamp}.png"

    x, y = lidar_pos
    gx, gy = goal_pos

    direct_dist = math.hypot(gx - x, gy - y)
    if direct_dist < 1e-9:
        print("Start and goal are the same.")
        return

    # Speed is calibrated so straight-line motion would take 240 s
    speed = direct_dist / DURATION_SEC
    step_dist = speed * DT

    current_angle = 0.0
    path_points = [(x, y)]
    world_scan_points = []
    reached = False
    crashed = False
    stuck = False

    t = 0.0
    max_sim_time = 2000.0   # safety cap so it cannot run forever

    print("Running moving simulation")
    print(f"Straight-line calibrated speed = {speed:.4f} m/s")

    with open(bin_path, "wb") as bf, open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["time_s", "x", "y", "heading_deg", "avoidance_active"])

        while t <= max_sim_time:
            move_heading, avoidance = choose_heading(x, y, gx, gy, step_dist)

            if move_heading is None:
                stuck = True
                print(f"No safe heading found at t={t:.2f}s near ({x:.2f}, {y:.2f})")
                writer.writerow([t, x, y, "", 1])
                break

            heading_rad = math.radians(move_heading)

            # generate one full scan cycle worth of packets
            num_packets_per_cycle = int(SAMPLE_FREQ / (POINTS_PER_PACKET * SCAN_FREQ))
            for _ in range(num_packets_per_cycle):
                pts = []
                start_angle_packet = current_angle

                for _ in range(POINTS_PER_PACKET):
                    local_angle = current_angle
                    world_angle = (move_heading + local_angle) % 360

                    d, intensity = get_distance(world_angle, x, y)
                    pts.append((d, intensity))

                    if d is not None:
                        ang = math.radians(world_angle)
                        wx = x + d * math.cos(ang)
                        wy = y + d * math.sin(ang)
                        world_scan_points.append((wx, wy))

                    current_angle = (current_angle + ANGLE_STEP) % 360

                bf.write(generate_packet(start_angle_packet, pts))

            # move after scanning
            nx = x + step_dist * math.cos(heading_rad)
            ny = y + step_dist * math.sin(heading_rad)

            if collides(nx, ny):
                crashed = True
                print(f"Collision at t={t:.2f}s near ({nx:.2f}, {ny:.2f})")
                writer.writerow([t, x, y, move_heading, int(avoidance)])
                break

            x, y = nx, ny
            path_points.append((x, y))
            writer.writerow([t, x, y, move_heading, int(avoidance)])

            if math.hypot(gx - x, gy - y) <= GOAL_TOLERANCE:
                reached = True
                print(f"Goal reached at t={t:.2f}s")
                break

            t += DT

    print(f"Saved LiDAR data to {bin_path}")
    print(f"Saved path log to {csv_path}")

    visualize_final_result(path_points, world_scan_points, reached, crashed, stuck, png_path)


# =========================
# FINAL VISUALIZATION
# =========================
def visualize_final_result(path_points, world_scan_points, reached, crashed, stuck, png_path):
    plt.figure(figsize=(9, 9))
    ax2 = plt.gca()

    # room
    ax2.add_patch(Rectangle((0, 0), ROOM_SIZE, ROOM_SIZE, fill=False, edgecolor='black', linewidth=2))

    # furniture
    for obj in objects:
        if obj["type"] == "rect":
            ax2.add_patch(Rectangle((obj["x"], obj["y"]), obj["w"], obj["h"], alpha=0.5))
        else:
            ax2.add_patch(Circle((obj["cx"], obj["cy"]), obj["r"], alpha=0.5))

    # lidar returns
    if world_scan_points:
        xs = [p[0] for p in world_scan_points]
        ys = [p[1] for p in world_scan_points]
        ax2.scatter(xs, ys, s=1, alpha=0.15, label="LiDAR returns")

    # path
    if path_points:
        px = [p[0] for p in path_points]
        py = [p[1] for p in path_points]
        ax2.plot(px, py, linewidth=2, label="Path")

    # start / goal
    if lidar_pos is not None:
        ax2.plot(lidar_pos[0], lidar_pos[1], 'rx', markersize=12, markeredgewidth=3, label="Start")
    if goal_pos is not None:
        ax2.plot(goal_pos[0], goal_pos[1], 'bx', markersize=12, markeredgewidth=3, label="Goal")

    if reached:
        status = "Reached goal"
    elif crashed:
        status = "Crashed"
    elif stuck:
        status = "Stuck"
    else:
        status = "Stopped"

    ax2.set_title(f"Moving LiDAR Simulation — {status}")
    ax2.set_xlim(0, ROOM_SIZE)
    ax2.set_ylim(0, ROOM_SIZE)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved final image to {png_path}")

    plt.show()


# =========================
# UI
# =========================
def redraw():
    ax.clear()
    ax.set_xlim(0, ROOM_SIZE)
    ax.set_ylim(0, ROOM_SIZE)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("r=rect, c=circle, l=start, g=goal, u=undo, enter=run moving sim")

    ax.add_patch(Rectangle((0, 0), ROOM_SIZE, ROOM_SIZE, fill=False, edgecolor='black', linewidth=2))

    for obj in objects:
        if obj["type"] == "rect":
            ax.add_patch(Rectangle((obj["x"], obj["y"]), obj["w"], obj["h"], alpha=0.5))
        else:
            ax.add_patch(Circle((obj["cx"], obj["cy"]), obj["r"], alpha=0.5))

    if lidar_pos is not None:
        ax.plot(lidar_pos[0], lidar_pos[1], 'rx', markersize=12, markeredgewidth=3)

    if goal_pos is not None:
        ax.plot(goal_pos[0], goal_pos[1], 'bx', markersize=12, markeredgewidth=3)

    fig.canvas.draw_idle()


def on_click(event):
    global temp_point, lidar_pos, goal_pos

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    x, y = event.xdata, event.ydata

    if mode == "lidar":
        lidar_pos = (x, y)
        redraw()
        return

    if mode == "goal":
        goal_pos = (x, y)
        redraw()
        return

    if temp_point is None:
        temp_point = (x, y)
        return

    x1, y1 = temp_point
    x2, y2 = x, y

    if mode == "rect":
        rx, ry = min(x1, x2), min(y1, y2)
        rw, rh = abs(x2 - x1), abs(y2 - y1)
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
    elif event.key == 'g':
        mode = "goal"
        temp_point = None
    elif event.key == 'u':
        if objects:
            objects.pop()
    elif event.key == 'enter':
        plt.close(fig)
        run_moving_simulation()
        return

    redraw()


def main():
    print("Interactive Moving LiDAR Simulator")
    print("Instructions:")
    print(" - r : rectangle furniture")
    print(" - c : circular furniture")
    print(" - l : place start / LiDAR")
    print(" - g : place goal")
    print(" - u : undo last furniture")
    print(" - enter : run moving simulation")

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
