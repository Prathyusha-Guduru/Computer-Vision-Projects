import cv2
import numpy as np
import time
# This library is for sending data to TouchDesigner/VCV Rack
from pythonosc import udp_client

# --- Configuration ---
# Update this with the path to your thermal video file.
# You can use a video file or a sequence of images.
VIDEO_SOURCE = 'fart_sample.mp4'
# Set the IP and port for your OSC client (e.g., TouchDesigner, VCV Rack)
# Check your software for the correct IP and port. 127.0.0.1 is for your local machine.
OSC_IP = "127.0.0.1"
OSC_PORT = 5005

# --- OSC Client Setup ---
# You can remove this part if you don't plan to use OSC
try:
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"OSC client initialized, sending data to {OSC_IP}:{OSC_PORT}")
except Exception as e:
    print(f"Could not initialize OSC client. Make sure you have python-osc installed and your IP/Port are correct. Error: {e}")
    osc_client = None

def detect_farts_and_diffuse(video_path):
    """
    Detects and analyzes heat blobs in a thermal video, simulating fart detection.
    Maps properties to OSC values for creative software.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Create a simple message box for information, as alerts are not supported in this environment
    print("\n--- Starting Thermal Analysis ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Restarting...")
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Convert the frame to grayscale to easily process thermal data
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # We assume that the heat sources (farts) are the brightest spots.
        # Use a threshold to isolate the hottest areas. The threshold value
        # may need to be adjusted depending on your video. Here we use a high value.
        # This creates a binary image where only the brightest pixels are white.
        _, thresh = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of the hot spots. Contours are outlines of detected objects.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a rectangle around each detected hot spot and get its properties
        for contour in contours:
            # Filter out very small areas that might be noise
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust this value to filter out small spots
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate the average temperature (brightness) of the detected area
                mask = np.zeros(gray_frame.shape, dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_val = cv2.mean(gray_frame, mask=mask)[0]

                # --- Map values to pitch, timbre, and effects ---
                # This is the core logic you'll use to drive your creative app.
                
                # Map area to a pitch value. Larger area = higher pitch.
                # Use a linear mapping for simplicity. Clamp values to a reasonable range.
                pitch_val = np.interp(area, [100, 5000], [0.1, 1.0])
                
                # Map mean temperature to a timbre or effect value. Hotter = more intense/distorted.
                timbre_val = np.interp(mean_val, [200, 255], [0.0, 1.0])

                # Map the diffusion rate (how fast the area is changing) to another effect.
                # This requires tracking the area over time, which we'll simulate here.
                # For a real implementation, you'd store the previous frame's area.
                # diffusion_rate = current_area - previous_area
                # For now, we'll just use the area directly as a proxy.
                diffusion_effect = np.interp(area, [100, 5000], [0.0, 1.0])
                
                # --- Send data via OSC ---
                if osc_client:
                    # Send a bundle of data with different addresses for each parameter
                    # You would set up your OSC client to listen to these addresses
                    osc_client.send_message("/fart/pitch", pitch_val)
                    osc_client.send_message("/fart/timbre", timbre_val)
                    osc_client.send_message("/fart/diffusion", diffusion_effect)
                    osc_client.send_message("/fart/area", float(area))
                    # Optionally, send coordinates for visual effects
                    osc_client.send_message("/fart/position", [float(x), float(y)])
                
                # --- Console Output for debugging ---
                print(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} | Hot spot detected:")
                print(f"  Area: {area:.2f} px")
                print(f"  Avg Temp: {mean_val:.2f} (0-255)")
                print(f"  OSC Values -> Pitch: {pitch_val:.2f}, Timbre: {timbre_val:.2f}")

                # Draw the contour on the original frame for visualization
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Draw a rectangle and display the mapped values on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area:.0f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Temp: {mean_val:.0f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow("Thermal Analysis", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("--- Analysis complete ---")

if __name__ == "__main__":
    detect_farts_and_diffuse(VIDEO_SOURCE)
