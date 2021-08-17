import ExperimentalImageStitcher as expim
import numpy as np
import cv2 
import ImageProcessor as ImgProc
from time import time 

def draw_matches(image_one, image_two):
    prev_keypoints, prev_descriptor = expim.generate_keypoints_and_descriptor(image_one)
    next_keypoints, next_descriptor = expim.generate_keypoints_and_descriptor(image_two)
    matches = expim.get_best_matching_keypoints(prev_keypoints, next_keypoints, prev_descriptor, next_descriptor)
    return cv2.drawMatches(image_one, prev_keypoints, image_two, next_keypoints, matches, image_one)

def lay_image_on_buffer(buffer, image, starting_offset):
    buffer[starting_offset[1]:(starting_offset[1] + len(image)), starting_offset[0]:(starting_offset[0] + len(image[0]))] = image
    return buffer

def extract_image_from_buffer(buffer, image, starting_offset):
    return buffer[starting_offset[1]:(starting_offset[1] + len(image)), starting_offset[0]:(starting_offset[0] + len(image[0]))]

def test_stitching(images):
    matching_points = expim.generate_matching_points(images)
    homographies = expim.generate_homographies(matching_points)
    ending_point, opposing_shift = expim.generate_ending_point_from_homographies(homographies) # TODO - Handle opposing shift
    
    starting_offset = np.array([abs(ending_point[0]) if ending_point[0] < 0 else 0, abs(ending_point[1]) if ending_point[1] < 0 else 0]) + opposing_shift
    output_size = np.flip(np.abs(ending_point[:-1])) + images[0].shape[:-1] + (opposing_shift * 2)

    rolling_image = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    input_image = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

    rolling_image = lay_image_on_buffer(rolling_image, images[0], starting_offset)
    for homography, matching_point, input in zip(homographies, matching_points, images[1:]):
        rolling_image = cv2.warpAffine(rolling_image, homography, np.flip(output_size))
        input, base_input = expim.erase_overlap_from_images(extract_image_from_buffer(rolling_image, input, starting_offset), input, matching_point)

        input_image = lay_image_on_buffer(input_image, input, starting_offset)
        rolling_image = lay_image_on_buffer(rolling_image, base_input, starting_offset)

        rolling_image = cv2.addWeighted(rolling_image, 1, input_image, 1, 0)
    
    return cv2.warpAffine(rolling_image, expim.get_homography(-opposing_shift), output_size)

def main():
    images = [cv2.imread(f'./DigPathSlideImages/William{i}.jpg') for i in range(1, 31)]
    start = time()
    cv2.imwrite('test_img.jpg', test_stitching(images))
    print(f'Time taken to complete operation: {time() - start}')

if __name__ == "__main__":
    main()