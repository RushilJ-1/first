import cv2
import numpy as np

def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_shapes = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)
        
        if vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
            detected_shapes.append((shape_name, cv2.contourArea(contour), (w, h), contour))
        elif vertices > 6:
            shape_name = "Circle"
            (x, y), radius = cv2.minEnclosingCircle(approx)
            detected_shapes.append((shape_name, cv2.contourArea(contour), (int(radius * 2), int(radius * 2)), contour))
        
    return detected_shapes

def detect_color(image, color_lower, color_upper, color_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def main():
    cap = cv2.VideoCapture(0)

    # Define the lower and upper bounds of the colors you want to detect
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    
    green_lower = np.array([25, 52, 72])
    green_upper = np.array([102, 255, 255])
    
    blue_lower = np.array([94, 80, 2])
    blue_upper = np.array([120, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        shapes = detect_shapes(frame)

        for shape, area, size, contour in shapes:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"{shape} - Area: {area:.2f} - Size: {size[0]}x{size[1]}", 
                        (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Detect red color in the frame
        red_result = detect_color(frame, red_lower, red_upper, "Red")

        # Detect green color in the frame
        green_result = detect_color(frame, green_lower, green_upper, "Green")

        # Detect blue color in the frame
        blue_result = detect_color(frame, blue_lower, blue_upper, "Blue")

        # Combine the results into a single image
        combined_result = cv2.addWeighted(red_result, 1, green_result, 1, 0)
        combined_result = cv2.addWeighted(combined_result, 1, blue_result, 1, 0)

        cv2.imshow("Shape and Color Detection", combined_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
