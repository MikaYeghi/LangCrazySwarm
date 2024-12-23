import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sns.set_theme()

# Functions
def get_values(VICON_DATA_DIR, RUN_NAME):
    # Load the data
    data_path = os.path.join(VICON_DATA_DIR, RUN_NAME)
    assert os.path.exists(data_path)
    data = pd.read_csv(data_path)

    # Extract X and Y coordinates
    xs = data['TX'].iloc[1:]
    ys = data['TY'].iloc[1:]
    assert len(xs) == len(ys)

    # Extract the values in an appropriate format
    x_values = [float(val) / 1000. for val in xs.tolist()]
    y_values = [float(val) / 1000. for val in ys.tolist()]

    return x_values, y_values

def plot_animation(x_values, y_values, output_path, fps=30, start_frame=None, end_frame=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    dx = max(x_values) - min(x_values)
    dy = max(y_values) - min(y_values)
    ax.set_xlim(min(x_values) - dx * 0.05, max(x_values) + dx * 0.05)
    ax.set_ylim(min(y_values) - dy * 0.05, max(y_values) + dy * 0.05)
    line, = ax.plot([], [], lw=2, label="Drone's Path")
    ax.grid(False)

    # Plot the room center with a larger red cross
    ax.plot([-1], [0], 'xr', markersize=10, label="Room Center")

    # Add legend
    ax.legend(loc="lower left")

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Cut the frames
    if start_frame is not None and end_frame is not None:
        x_values = x_values[start_frame:end_frame]
        y_values = y_values[start_frame:end_frame]

    # Frames per second and total duration
    frames = len(x_values)

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        return line,

    # Function to update the plot at each frame
    def update(frame):
        # Determine the range of points to show
        progress = frame / frames
        if frame % 10 == 0:
            print(f"Progress: {round(progress * 100, 2)}%")
        num_points = int(progress * len(x_values))
        line.set_data(x_values[:num_points], y_values[:num_points])
        return line,

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=1000/fps
    )

    # Save the animation
    print("Saving the animation...")
    # output_path = os.path.join(save_dir, "trajectory_animation.mp4")
    ani.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    print(f"Animation saved to {output_path}")
    plt.close()

# Configs
VICON_DATA_DIR = "results/Vicon_Data"
RUN_NAME = "LangCrazySwarm_LastDay_Lissajous-02.csv"
SAVE_DIR = "plots"
FPS = 30
SHAPE_FRAMES = {
    "ellipse": {
        "start_frame": 867,
        "end_frame": 1560
    },
    "heart": {
        "start_frame": 1100,
        "end_frame": 1820
    },
    "Lissajous": {
        "start_frame": 2160,
        "end_frame": 2860
    }
}
SHAPE = "Lissajous" # One of: [heart, ellipse, Lissajous]

# Create paths and save directories
save_dir = os.path.join(SAVE_DIR, RUN_NAME.split('.')[0])
os.makedirs(save_dir, exist_ok=True)
print(f"Saving the plots to {save_dir}")

# Extract the X and Y values
x_values, y_values = get_values(VICON_DATA_DIR, RUN_NAME)

# Save the full animation
output_path = os.path.join(save_dir, "full_trajectory_animation.mp4")
plot_animation(
    x_values,
    y_values,
    output_path, 
    fps=FPS,
)

# Trim the coordinates to plot the shape only
start_frame = SHAPE_FRAMES[SHAPE]["start_frame"]
end_frame = SHAPE_FRAMES[SHAPE]["end_frame"]
output_path = os.path.join(save_dir, "shape_trajectory_animation.mp4")
plot_animation(
    x_values,
    y_values,
    output_path, 
    fps=FPS,
    start_frame=start_frame, 
    end_frame=end_frame, 
)