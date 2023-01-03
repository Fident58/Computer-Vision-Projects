import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Loads an image from the specified path.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
    return img

def convert_to_grayscale(image):
    """
    Converts a color image to grayscale.
    """
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """
    Applies Gaussian blur to an image.
    """
    if image is None:
        return None
    return cv2.GaussianBlur(image, kernel_size, sigmaX)

def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """
    Detects edges in an image using the Canny algorithm.
    """
    if image is None:
        return None
    return cv2.Canny(image, low_threshold, high_threshold)

def save_image(image, output_path):
    """
    Saves the processed image to the specified path.
    """
    if image is None:
        print("Error: No image to save.")
        return False
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")
    return True

if __name__ == "__main__":
    # Create a dummy image for demonstration if it doesn't exist
    dummy_image_path = "dummy_image.png"
    if not os.path.exists(dummy_image_path):
        dummy_image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Hello CV!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(dummy_image_path, dummy_image)
        print(f"Created dummy image: {dummy_image_path}")

    input_image_path = dummy_image_path
    output_dir = "./processed_images"
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    original_image = load_image(input_image_path)

    if original_image is not None:
        # Convert to grayscale
        gray_image = convert_to_grayscale(original_image)
        save_image(gray_image, os.path.join(output_dir, "grayscale_image.png"))

        # Apply Gaussian blur
        blurred_image = apply_gaussian_blur(original_image)
        save_image(blurred_image, os.path.join(output_dir, "blurred_image.png"))

        # Detect edges
        edges_image = detect_edges_canny(gray_image)
        save_image(edges_image, os.path.join(output_dir, "edges_image.png"))

        print("\nImage processing demonstration complete.")
