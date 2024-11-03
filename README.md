# **Self-Driving-Car (High School Project)**
In high school, me and my friend participated in the CBSE Science Exhibition, where our project reached the final round. We developed a neural network for object detection using the TensorFlow API and the SSD MobileNet model. While I did not develop the neural network from scratch, I trained and modified it by adjusting various settings. The model recognized custom traffic signs and toy figures as pedestrians, trained and run on an NVIDIA GTX 1650 GPU.

To estimate object distances, we applied the pinhole camera principle. By utilizing objects of known height, we calculated the ratio of pixel height to actual height, enabling us to measure the exact distance to detected objects based on their pixel dimensions.
![distanceCalc](https://github.com/user-attachments/assets/896a9a2a-dc50-4eb0-9f5e-629469f224b5)


For lane detection, we implemented a series of steps using the OpenCV library:

> Converted the image from RGB to HSV to separate intensity from color.
> Extracted the lane line colors, applying a color chart for accurate extraction.
> Applied Gaussian Blur to reduce noise.
> Implemented Canny Edge Detection to identify edges.
> Cropped the image to focus on the lower half, where lane lines appear.
> Used Hough Line Transform to detect line segments.
> Fitted a first-degree polynomial to the detected lines and averaged the left and right lanes.
> Averaged the endpoints of the two lines for smoother lane following.

![laneDet](https://github.com/user-attachments/assets/20a16dcd-23dd-45c2-a562-49d30f054c1b)


Hardware Components

> Chassis: Made from an old wooden frame.
> Camera: A phone camera serves as the car's eyes.
> Motors: Four motors control the wheels, using relative velocity for turning.
> Control System: An Arduino Uno R3 with HC-05 Bluetooth module drives the motors via two L293D motor drivers, powered by a 9V battery (Arduino) and 12V (motor drivers).

![Capture](https://github.com/user-attachments/assets/6617c4e8-a56f-4abc-8b57-1e4c03db0141)

The camera streams video to a computer, which performs model inference and lane detection, sending commands to the Arduino via Bluetooth for car navigation.

![obdetTest2](https://github.com/user-attachments/assets/88571b6b-3b15-4168-b904-ea2a5c253ed6)


For the car's movement algorithm, we ensured gradual steering towards the detected average line instead of abruptly changing angles, which minimized issues caused by noise in lane detection.
