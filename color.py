import cv2
import numpy as np

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

        # Detect red color in the frame
        red_result = detect_color(frame, red_lower, red_upper, "Red")
        color_name = "Red"

        # Detect green color in the frame
        green_result = detect_color(frame, green_lower, green_upper, "Green")
        color_name = "Green"

        # Detect blue color in the frame
        blue_result = detect_color(frame, blue_lower, blue_upper, "Blue")
        color_name = "Blue"

        # Combine the results into a single image
        combined_result = cv2.addWeighted(red_result, 1, green_result, 1, 0)
        combined_result = cv2.addWeighted(combined_result, 1, blue_result, 1, 0)

        # Convert the combined result to grayscale for contour detection
        gray = cv2.cvtColor(combined_result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a rectangle or circle near the object's contour and display the detected color within that shape
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Adjust this threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                color_area = combined_result[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, "Detected Color", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, "Color: " + color_name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the original frame
        cv2.imshow("Color Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
