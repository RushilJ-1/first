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
        
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif vertices == 5:
            shape_name = "Pentagon"
        elif vertices == 6:
            shape_name = "Hexagon"
        else:
            shape_name = "Circle"
        
        area = cv2.contourArea(contour)
        size = cv2.boundingRect(contour)[2:]  # width and height
        detected_shapes.append((shape_name, area, size, contour))
        
    return detected_shapes

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    shapes = detect_shapes(frame)
    
    for shape_info in shapes:
        shape, area, size, contour = shape_info
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"{shape} - Area: {area:.2f} - Size: {size[0]}x{size[1]}", 
                    (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow('Shapes Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
