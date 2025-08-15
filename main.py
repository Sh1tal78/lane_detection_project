import cv2
import numpy as np

# ---------- Lane Detection Functions ----------
def preprocess_frame(frame):
    """Convert to grayscale, blur, and detect edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    """Mask the image to keep only the road region."""
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define a triangular polygon for ROI
    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2 + 50, height // 2),
        (width // 2 - 50, height // 2)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def detect_lines(roi):
    """Detect lane lines using Hough Transform."""
    return cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

def average_slope_intercept(frame, lines):
    """Average lines into a single left and right line."""
    left_lines, right_lines = [], []
    left_weights, right_weights = [], []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:  # Avoid division by zero
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if slope < 0:  # Left lane
            left_lines.append((slope, intercept))
            left_weights.append(length)
        else:          # Right lane
            right_lines.append((slope, intercept))
            right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None

    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """Convert slope & intercept into pixel points."""
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

def draw_lines(frame, lines, color=(0, 255, 0), thickness=10):
    """Draw left and right lane lines."""
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def lane_departure_warning(frame, left_line, right_line):
    """Display warning if vehicle deviates from lane center."""
    height, width, _ = frame.shape
    if left_line and right_line:
        lane_center = (left_line[1][0] + right_line[1][0]) // 2
        frame_center = width // 2
        deviation = lane_center - frame_center

        if abs(deviation) > 50:  # Pixel threshold for warning
            cv2.putText(frame, "WARNING: Lane Departure!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
    return frame

# ---------- Main Program ----------
cap = cv2.VideoCapture("highway.mp4")
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess_frame(frame)
    roi = region_of_interest(edges)
    lines = detect_lines(roi)
    left_lane, right_lane = average_slope_intercept(frame, lines)

    # Define points for drawing
    y1 = frame.shape[0]
    y2 = int(y1 * 0.6)
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    # Draw lanes
    frame_with_lanes = draw_lines(frame, [left_line, right_line])

    # Add lane departure warning
    frame_with_warning = lane_departure_warning(frame_with_lanes, left_line, right_line)

    # Display output
    cv2.imshow("Lane Detection", frame_with_warning)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
