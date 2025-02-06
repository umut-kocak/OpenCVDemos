"""
"""
from pathlib import Path
import cv2


from utils.base_video_demo import BaseVideoDemo
from utils.video_writer import VideoWriter
import utils.image_styler as image_styler

class StyleTransferDemo(BaseVideoDemo):
    """Demo for detecting faces in video frames."""

    def __init__(self):
        super().__init__()

        # Load a custom background image
        #_style_image_path = self.get_assets_folder() / self.settings.demo.style_image
        #self._style_image = cv2.imread(_style_image_path)
        #if self._style_image is None:
        #    logging.error("No stlye image found: %s", _style_image_path)
        #self._style_image = cv2.cvtColor(self._style_image, cv2.COLOR_BGR2RGB)
        #self._style_image = cv2.convertScaleAbs(self._style_image, alpha=0.8, beta=0)  # Reduce brightness

        self._output_index = None
        if self.settings.demo.save_frames:
            self._output_index = 0

        v_settings = self.settings.video.output
        self._video_writer = VideoWriter(self.get_output_folder() / v_settings.file_name, fourcc='mp4v',
            fps=self._video_manager.fps, frame_size=(self._video_manager.width, self._video_manager.height))

    def process_frame(self, frame):
        """
        """
        #styled_image = image_styler.style_transfer(frame.image, self._style_image)
        #styled_image = cv2.cvtColor(styled_image, cv2.COLOR_RGB2BGR)
        #styled_image = image_styler.fast_neural_style_transfer(frame.image, self.get_assets_folder()/ "models" / "rain_princess.pth")
        styled_image = image_styler.fast_neural_style_transfer(frame.image, self.get_asset_path( "models/rain_princess.pth") )
        #styled_image = frame.image

        #if self._output_index is not None:
        #    output_path = self.get_output_folder() / ('stlyed-' + str(self._output_index) + '.jpg')
        #    cv2.imwrite(output_path, styled_image)
        #    self._output_index += 1
        #self._video_writer.write(styled_image)

        frame.image = styled_image
        return frame

    def cleanup(self):
        """Perform cleanup tasks when the demo is stopped."""
        super().cleanup()
        self._video_writer.release()

    def get_window_name(self):
        """Return the name of the demo window."""
        return "Style Transfer Demo"


if __name__ == "__main__":
    demo = StyleTransferDemo()
    demo.run()
