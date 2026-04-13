import cv2
import os
import pickle
import argparse
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ROIDrawer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.image_copy = self.image.copy()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.roi = None
        self.window_name = "Draw ROI (Press 's' to save, 'r' to reset, 'q' to quit)"

    def draw_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.image_copy = self.image.copy()
            cv2.rectangle(self.image_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi = (min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y))
            cv2.rectangle(self.image_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            logger.info(f"ROI selected: {self.roi}")

    def draw_roi(self, output_path):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_callback)

        while True:
            cv2.imshow(self.window_name, self.image_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and self.roi is not None:
                with open(output_path, 'wb') as f:
                    pickle.dump(self.roi, f)
                logger.info(f"Saved ROI to: {output_path}")
                break
            elif key == ord('r'):
                self.image_copy = self.image.copy()
                self.roi = None
                logger.info("Reset ROI")
            elif key == ord('q'):
                logger.info("Quit without saving")
                break

        cv2.destroyAllWindows()
        return self.roi

def main():
    parser = argparse.ArgumentParser(description="Draw ROI on an image and save coordinates")
    parser.add_argument('--image_path', type=str, required=True, help='Path to sample image or video frame')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save ROI coordinates (e.g., dime_roi.pkl)')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        logger.error(f"Image path does not exist: {args.image_path}")
        return

    drawer = ROIDrawer(args.image_path)
    roi = drawer.draw_roi(args.output_path)
    if roi:
        logger.info(f"Final ROI: {roi}")
    else:
        logger.warning("No ROI saved")

if __name__ == "__main__":
    main()