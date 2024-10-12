import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd

# Assuming you already have the 'advertising' DataFrame
advertising = pd.read_csv("advertising.csv")
# Calculate the mean of Sales
mean_sales = advertising['Sales'].mean()

# Linear regression line (y = mx + b)
# Calculate slope (m) and intercept (b)
m, b = np.polyfit(advertising['TV'], advertising['Sales'], 1)

# Setup figure and scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(advertising['TV'], advertising['Sales'], color='orange')
ax.set_title("TV Advertising Spend vs Sales with Animated Line Transition")
ax.set_xlabel("TV Advertising Spend ($)")
ax.set_ylabel("Sales ($)")
ax.grid(True)

# Initialize the line with the mean
line, = ax.plot(advertising['TV'], [mean_sales]*len(advertising['TV']), color='blue')

# Define animation function
def animate(i):
    slope = (m / 100) * i  # Incrementally increase the slope
    intercept = mean_sales + (b - mean_sales) * (i / 100)  # Adjust the intercept gradually
    line.set_ydata(slope * advertising['TV'] + intercept)  # Update the line data
    return line,

# Animate over 100 frames
ani = FuncAnimation(fig, animate, frames=100, interval=50)

# Set up the writer for MP4
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as an MP4 file
ani.save("linear_regression_animation.mp4", writer=writer)

# Show plot (optional, not necessary for saving)
plt.show()
