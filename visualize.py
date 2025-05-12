import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.dates as mdates

def visualize_trajectory(data_points):
    """
    Visualize the trajectory of an object based on (x, y, time) data points.
    
    Args:
        data_points: List of tuples (x, y, time_str) where time_str is in format 'HH:MM:SS.microseconds'
    """
    # Extract coordinates and times
    x_coords = [point[0] for point in data_points]
    y_coords = [point[1] for point in data_points]
    time_strs = [point[2] for point in data_points]
    
    # Convert time strings to datetime objects
    base_date = "2025-05-11 "  # Using current date as base
    times = [datetime.strptime(base_date + t, "%Y-%m-%d %H:%M:%S.%f") for t in time_strs]
    
    # Calculate time differences from start in seconds
    start_time = times[0]
    time_diffs = [(t - start_time).total_seconds() for t in times]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set up the trajectory plot (left subplot)
    ax1.set_title('Object Trajectory')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Plot the full trajectory path
    trajectory_line, = ax1.plot(x_coords, y_coords, 'b-', alpha=0.3, label='Full Path')
    
    # Point that will be animated - initialize with first point
    point, = ax1.plot([x_coords[0]], [y_coords[0]], 'ro', markersize=8, label='Current Position')
    
    # Add text annotation for time and position
    position_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Setting reasonable axis limits
    margin = 5
    ax1.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax1.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax1.grid(True)
    ax1.legend(loc='lower right')
    
    # Set up the position vs time plot (right subplot)
    ax2.set_title('Position vs Time')
    ax2.set_xlabel('Time (seconds from start)')
    ax2.set_ylabel('Position')
    
    # Create empty lines for x and y positions over time
    x_time_line, = ax2.plot([], [], 'r-', label='X Position')
    y_time_line, = ax2.plot([], [], 'g-', label='Y Position')
    
    ax2.set_xlim(0, max(time_diffs) * 1.1)
    ax2.set_ylim(min(min(x_coords), min(y_coords)) - margin, 
                max(max(x_coords), max(y_coords)) + margin)
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    # Calculate speed between points
    speeds = []
    for i in range(1, len(data_points)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        dt = time_diffs[i] - time_diffs[i-1]
        if dt > 0:
            speed = np.sqrt(dx**2 + dy**2) / dt
        else:
            speed = 0
        speeds.append(speed)
    
    # Add average speed to the plot
    if speeds:
        avg_speed = np.mean(speeds)
        speed_text = f"Average Speed: {avg_speed:.2f} units/second"
        ax1.text(0.02, 0.02, speed_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Animation update function
    def update(frame):
        # Update the trajectory point position
        if frame < len(x_coords):
            point.set_data([x_coords[frame]], [y_coords[frame]])  # Pass as lists
            
            # Update the current position and time text
            time_info = f"Time: {time_diffs[frame]:.2f}s"
            pos_info = f"Position: ({x_coords[frame]:.2f}, {y_coords[frame]:.2f})"
            position_text.set_text(f"{time_info}\n{pos_info}")
            
            # Update position vs time plot
            x_time_line.set_data(time_diffs[:frame+1], x_coords[:frame+1])
            y_time_line.set_data(time_diffs[:frame+1], y_coords[:frame+1])
            
        return point, position_text, x_time_line, y_time_line
    
    # Create animation - use a simpler approach
    frames = len(x_coords)
    ani = FuncAnimation(fig, update, frames=range(frames), interval=200, blit=True, repeat=False)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
    # Print additional information about the trajectory
    print(f"Total trajectory distance: {calculate_total_distance(x_coords, y_coords):.2f} units")
    print(f"Total time: {time_diffs[-1]:.2f} seconds")
    if speeds:
        print(f"Maximum speed: {max(speeds):.2f} units/second")
        print(f"Minimum speed: {min(speeds):.2f} units/second")
    
    # Return the animation object to prevent garbage collection
    return ani

def calculate_total_distance(x_coords, y_coords):
    """Calculate the total distance traveled along the trajectory."""
    total_distance = 0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        total_distance += distance
    return total_distance

# Example usage with your data
if __name__ == "__main__":
    data_points = [
        (-1254.2237369101495, -205.76573361363262, '23:23:06.041997'),
        (-1251.1982442419976, -205.90137551631778, '23:23:06.088884'),
        (-1248.0006958525628, -205.54404409136623, '23:23:06.135752'),
        (-1245.4993969481438, -204.60765035357326, '23:23:06.182629'),
        (-1240.3846012633294, -203.07262010779232, '23:23:06.245124')
    ]
    
    # Run the visualization
    ani = visualize_trajectory(data_points)