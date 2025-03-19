# Use an official Python image as base
FROM python:3.13

# Set the working directory
WORKDIR /app

# Install necessary system libraries for Qt and X11
RUN apt-get update && apt-get install -y \
    python3-pyqt5 \
    pyqt5-dev-tools \
    libx11-xcb1 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxcb-keysyms1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-render-util0 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Set environment variable to use X11
ENV QT_QPA_PLATFORM=xcb

# Run the application
CMD ["python", "app.py"]
