import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_images(org_path, edited_path):
    org = cv2.imread(org_path)
    edited = cv2.imread(edited_path)
    return org, edited

def dilate_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = np.zeros_like(mask)

    for _ in range(iterations):
        for i in range(mask.shape[0] - kernel_size + 1):
            for j in range(mask.shape[1] - kernel_size + 1):
                if mask[i, j] > 0:
                    dilated[i:i+kernel_size, j:j+kernel_size] = np.maximum(
                        dilated[i:i+kernel_size, j:j+kernel_size], kernel
                    )

    return (dilated > 0).astype(np.uint8) * 255
def label_regions(mask):
    labeled_mask = np.zeros_like(mask)
    current_label = 1
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 255 and labeled_mask[y, x] == 0:
                stack = [(y, x)]
                while stack:
                    current_y, current_x = stack.pop()
                    labeled_mask[current_y, current_x] = current_label
                    neighbors = [(current_y - 1, current_x),
                                 (current_y + 1, current_x),
                                 (current_y, current_x - 1),
                                 (current_y, current_x + 1)]
                    stack.extend((ny, nx) for ny, nx in neighbors
                                 if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                 and mask[ny, nx] == 255 and labeled_mask[ny, nx] == 0)
                current_label += 1
    return labeled_mask

def find_contours_and_bbox(mask):
    labeled_mask = label_regions(mask)
    contours = []
    for label_value in range(1, np.max(labeled_mask) + 1):
        ys, xs = np.where(labeled_mask == label_value)
        if len(xs) > 0 and len(ys) > 0:  # Skip empty contours
            contours.append(np.column_stack((xs, ys)))
    return contours


def create_mask(org, edited, threshold=100):
    diff = np.abs(org.astype(np.int32) - edited.astype(np.int32))
    diff_sum = np.sum(diff, axis=-1)

    # Convert to binary mask
    mask = (diff_sum > threshold).astype(np.uint8) * 255

    # Dilate the mask
    dilated = dilate_mask(mask)

    # Apply the mask to the edited image
    masked = edited.copy()
    masked[dilated == 0] = 0

    return dilated, masked

def calculate_contour_area(contour):
    return 0.5 * np.abs(np.dot(contour[:, 0], np.roll(contour[:, 1], 1)) -
                        np.dot(np.roll(contour[:, 0], 1), contour[:, 1]))

def calculate_bounding_rect(contour):
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    return x_min, y_min, x_max - x_min, y_max - y_min

def draw_bounding_boxes(edited, contours, min_area=100):
    for con in contours:
        area = calculate_contour_area(con)

        if area > min_area:
            x, y, w, h = calculate_bounding_rect(con)
            rect_patch = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=(0, 0, 1), facecolor='none')
            plt.gca().add_patch(rect_patch)

    return edited

def crop_and_display_largest_objects(image, contours, num_objects=3, min_area=100):
    sorted_contours = sorted(contours, key=lambda x: calculate_contour_area(x), reverse=True)[:num_objects]

    for i, con in enumerate(sorted_contours):
        area = calculate_contour_area(con)

        if area > min_area:
            x, y, w, h = calculate_bounding_rect(con)
            cropped_object = image[y:y + h, x:x + w]
            display_image(cropped_object)

def bgr_to_rgb(image):
    return image[..., ::-1]  

def display_image(image):
    if image.shape[-1] == 4:
        image = image[..., :3]

    image_rgb = bgr_to_rgb(image)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

org_image_path = 'org.jpg'
edited_image_path = 'edited.jpg'
org, edited = load_images(org_image_path, edited_image_path)

# Create mask, find contours, and draw bounding boxes
dilated, masked = create_mask(org, edited)
contours = find_contours_and_bbox(dilated)
result_image = draw_bounding_boxes(edited.copy(), contours, min_area=1)

# Remove background from the original image using the dilated mask
org_no_bg = edited.copy()
org_no_bg[dilated == 0] = 0

display_image(result_image)
crop_and_display_largest_objects(org_no_bg, contours, num_objects=3, min_area=1)
