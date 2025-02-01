import cv2

from utils.logger import logger

class VideoWriter:
    def __init__(self, filename, fourcc='MJPG', fps=30, frame_size=(640, 480)):
        """
        Initializes the VideoWriter.
        
        :param filename: Output video file name.
        :param fourcc: FourCC codec code (default: 'MJPG').
        :param fps: Frames per second (default: 30).
        :param frame_size: Frame size (default: (640, 480)).
        """
        # Create the VideoWriter object
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.out = cv2.VideoWriter(self.filename, self.fourcc, self.fps, self.frame_size)
        print(filename)
        print(frame_size)
        print(fps)

        if not self.out.isOpened():
            logger.error("Failed to open VideoWriter with filename: %s", self.filename)
            exit(1)
    
    def write(self, frame):
        """
        Write a single frame to the video.
        
        :param frame: A single frame (numpy array) to write.
        """
        if self.out.isOpened():
            self.out.write(frame)
        else:
            logger.error("VideoWriter is not opened. Can not write the frame.")
    
    def release(self):
        """
        Release the VideoWriter and close the output file.
        """
        if self.out.isOpened():
            self.out.release()
            logger.info("Video saved to %s.", self.filename)
            logger.debug("VideoWriter is already released")
        else:
            logger.debug("VideoWriter is already released.")

# Example usage:
if __name__ == "__main__":
    # Set up video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        logger.error("Failed to open the webcam.")
        exit(1)
    
    # Create VideoWriter instance
    video_writer = VideoWriter('output_video.avi', fourcc='MJPG', fps=30, frame_size=(640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to match the frame size of VideoWriter
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Write the frame to the video file
        video_writer.write(frame_resized)
        
        # Display the frame
        cv2.imshow('Webcam', frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()