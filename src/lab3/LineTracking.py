import cv2
import numpy as np
from cv2.typing import Point


def detectLine(
        frame: np.ndarray,
        des_color: np.ndarray,
        thresh: float = 100.0,
        gradient_threshold: float = 15.0,   
        min_area_px: int = 20,              
        max_area_fraction: float = 0.6,     
        morph_kernel: int = 1):
    """
    Process the given frame to detect and track the center of a white line.
    
    Args:
        frame (numpy.ndarray): The input frame from the webcam.
    
    Returns:
        lineCenter: A number between [-1, 1] denoting where the center of the line is relative to the frame.
        newFrame: Processed frame with the detected line marked using cv2.rectangle() and center marked using cv2.circle().
    """
    
    # may be unnecessary
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = frame.shape[:2]

    # sobel in both directions
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3) # claude suggested this kernel size
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    
    # overall gradient/edges
    grad_mag = cv2.magnitude(gx, gy)
    edges = (grad_mag > gradient_threshold).astype(np.uint8) * 255

    # thicken edges
    k = max(1, morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # this does it without thickening edges
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    regions = cv2.bitwise_not(edges_closed)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    max_area = max_area_fraction * width * height

    
    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]
        if area < min_area_px or area > max_area:
            continue
        avg_color = frame[labels == i].mean(axis=0)
        if np.linalg.norm(avg_color - des_color) < thresh:
            centerpt = (x + w/2, y + h/2)
            lineCenter = (centerpt[0] - width/2) / (width/2)
            newFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            newFrame = cv2.circle(newFrame, (width//2, height//2), 2, (0, 0, 255))
            return lineCenter, newFrame

    return None, frame    


def main():
    cam = cv2.VideoCapture(0)  # Open webcam

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        lineCenter, newFrame = detectLine(frame, (255, 255, 255))

        print(f"LineCenter: {lineCenter}")
        cv2.imshow("frame",newFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
