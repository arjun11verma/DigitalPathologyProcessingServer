import cv2
import numpy as np
import sys 
from edge import Edge
from enum import Enum

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
every_other = 0

class edges(Enum):
    top_edge = 0,
    right_edge = 1,
    bottom_edge = 2,
    left_edge = 3

def display_image(image):
    cv2.imshow('dst', image.astype('uint8'))
    cv2.waitKey(0)

def draw_keypoints(image, keypoints):
    cv2.drawKeypoints(image, keypoints, image)
    return image 

def generate_edges(image, lower_threshold, upper_threshold, view_image=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(image, lower_threshold, upper_threshold)
    
    if view_image: display_image(canny_edges)
    return canny_edges

def generate_harris_corners(image, threshold, view_image=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray, 2, 5, 0.04)
    harris_corner_max_threshold = threshold * harris_corners.max()
    harris_corners = np.argwhere(harris_corners > harris_corner_max_threshold)

    return harris_corners

def generate_keypoints_from_features(features):
    return [cv2.KeyPoint(float(point[1]), float(point[0]), 1) for point in features]

def generate_SIFT_descriptor(image, view_image=False):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    image = cv2.drawKeypoints(image, keypoints, image)

    if view_image:
        display_image(image)

    return keypoints

def gaussian_smooth(image, sigma_x, sigma_y=None):
    if not sigma_y: sigma_y = sigma_x
    return cv2.GaussianBlur(image, (5, 5), sigmaX=sigma_x, sigmaY=sigma_y)

def generate_brief_descriptor(image, keypoints):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = gaussian_smooth(image, 1.73)
    return brief.compute(image, keypoints)

def generate_keypoints_and_descriptor(image):
    keypoints = generate_keypoints_from_features(generate_harris_corners(image, 0.05))
    return generate_brief_descriptor(image, keypoints)

def get_hamming_distance_from_BRIEF(descriptor_one, descriptor_two):
    hamming_distance = 0
    for i in range(len(descriptor_one)):
        for j in range(8):
            if (descriptor_one[i] & (1 << j) != descriptor_two[i] & (1 << j)): hamming_distance += 1
    return hamming_distance

def get_matching_points(keypoints_one, keypoints_two, descriptor_one, descriptor_two):
    matches = bf_matcher.match(descriptor_one, descriptor_two)
    matches = sorted(matches, key = lambda match : match.distance)
    matches = matches[:10] if len(matches) > 10 else matches
    matches = [{"First": keypoints_one[match.queryIdx].pt, "Second": keypoints_two[match.trainIdx].pt} for match in matches]

    return matches

def get_best_matching_point(keypoints_one, keypoints_two, descriptor_one, descriptor_two):
    matches = bf_matcher.match(descriptor_one, descriptor_two)

    best_match = min(matches, key= lambda match : match.distance)
    first, second = keypoints_one[best_match.queryIdx].pt, keypoints_two[best_match.trainIdx].pt
    first, second = (int(first[0]), int(first[1])), (int(second[0]), int(second[1]))
    best_match = {"First": first, "Second": second}

    return best_match

def get_best_matching_keypoints(keypoints_one, keypoints_two, descriptor_one, descriptor_two):
    matches = bf_matcher.match(descriptor_one, descriptor_two)
    matches = sorted(matches, key= lambda match : match.distance)[:30]
    return [keypoints_one[m.queryIdx] for m in matches], [keypoints_one[m.trainIdx] for m in matches]

def calculate_image_offset(matching_point):
    x_offset = (matching_point['Second'][0] - matching_point['First'][0])
    y_offset = (matching_point['Second'][1] - matching_point['First'][1])
    return np.array([x_offset, y_offset])

def get_homography(matching_offsets):
    return np.array([
         [1, 0, matching_offsets[0]],
         [0, 1, matching_offsets[1]]
    ], dtype=np.float)

def generate_matching_points(images):
    matching_points = []
    prev_keypoints, prev_descriptor = generate_keypoints_and_descriptor(images[0])

    for image in images[1:]:
        next_keypoints, next_descriptor = generate_keypoints_and_descriptor(image)
        best_matching_point = get_best_matching_point(prev_keypoints, next_keypoints, prev_descriptor, next_descriptor)
        matching_points.append(best_matching_point)
        prev_keypoints, prev_descriptor = next_keypoints, next_descriptor
    
    return matching_points

def generate_homographies(matching_points):
    return [get_homography(calculate_image_offset(matching_point)) for matching_point in matching_points]

def generate_ending_point_from_homographies(homographies):
    ending_point = np.array([0, 0, 1], dtype=np.float)
    opposing_shift = np.array([1000, 1000]) # Buffer for up and down behavior - TODO Handle how to generate this correctly programmatically 
    for homography in homographies:
        ending_point = homography @ ending_point
        ending_point = np.append(ending_point, 1.0)
    return ending_point.astype('int'), opposing_shift.astype('int')

def erase_overlap_from_images(base_image, input_image, matching_point):
    overlap_mask = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    threshold, overlap_mask = cv2.threshold(overlap_mask, 1, 255, cv2.THRESH_BINARY)

    carve_mask = cv2.bitwise_not(overlap_mask)

    input_image = cv2.bitwise_and(input_image, input_image, mask=carve_mask)
    base_image = cv2.bitwise_and(base_image, base_image, mask=cv2.bitwise_not(carve_mask))

    return input_image, base_image

def cost_function(input_1, input_2, base_1, base_2):
    return 

def generate_graph_cut_graph(input_image, base_image):
    edge_list = []

    for i in range(base_image):
        for j in range(base_image[0]):
            if (base_image[i][j] == [0, 0, 0]):
                edge_list.append(Edge([i, j], [i + 1, j], sys.maxsize))
                edge_list.append(Edge([i, j], [i, j + 1], sys.maxsize))
            else:
                edge_list.append(Edge([i, j], [i + 1, j], cost_function(input_image[i][j], input_image[i+1][j], base_image[i][j], base_image[i+1][j])))
                edge_list.append(Edge([i, j], [i, j + 1], cost_function(input_image[i][j], input_image[i][j+1], base_image[i][j], base_image[i][j+1])))


def seam_carve(image_gradient, gradient_mask):
    energy_table = (cv2.bitwise_not(gradient_mask)).astype('uint64')

    carve_mask = np.full(image_gradient.shape, 255)
    top_edge, right_edge, bottom_edge, left_edge = determine_ending_edges(gradient_mask)

    image_segment = image_gradient[min(right_edge['start'], left_edge['start']):max(right_edge['end'], left_edge['end']), min(top_edge['start'], bottom_edge['start']):max(top_edge['end'], bottom_edge['end'])]
    energy_table = energy_table[min(right_edge['start'], left_edge['start']):max(right_edge['end'], left_edge['end']), min(top_edge['start'], bottom_edge['start']):max(top_edge['end'], bottom_edge['end'])]

    carve = get_carve_mask(energy_table, image_segment, top_edge, right_edge, bottom_edge, left_edge)
    carve_mask[min(right_edge['start'], left_edge['start']):max(right_edge['end'], left_edge['end']), min(top_edge['start'], bottom_edge['start']):max(top_edge['end'], bottom_edge['end'])] = carve if len(carve) != 0 else gradient_mask[min(right_edge['start'], left_edge['start']):max(right_edge['end'], left_edge['end']), min(top_edge['start'], bottom_edge['start']):max(top_edge['end'], bottom_edge['end'])]

    return carve_mask.astype('uint8')

def get_carve_mask(energy_table, image_gradient, top_edge, right_edge, bottom_edge, left_edge):
    energy_table_v = fill_energy_table(np.copy(energy_table), image_gradient)
    energy_table_h = fill_energy_table(np.copy(energy_table), image_gradient, horizontal=True)

    energy_table_diag = energy_table_v + energy_table_h
    left_corner_diag_traversal = traverse_energy_table_diag(energy_table_diag, edges.right_edge)

    return left_corner_diag_traversal # determine whether or not to and or or based off of what will overlap what
    # ie - you could have a situation wherein the left fill of the vertical traversal overwrites the bottom gap of the horizonatal traversal. Only what is shared shuold be kept in that situation. In others, both should be kept. 

class Point:
    def __init__(self, position, value):
        self.value = value
        self.position = position 

def traverse_energy_table_diag(energy_table, starting_edge):
    right_edges = energy_table[:, -1]
    carve_mask = np.zeros(energy_table.shape).astype('uint8')
    starting_point = np.array([len(energy_table) - 1, len(energy_table[0]) - 1])

    carve_mask[starting_point[0], :starting_point[1]] = 255
    while (starting_point[0] > 0 and starting_point[1] > 0):
        energy_table[starting_point[0], starting_point[1]] = sys.maxsize
        point_list = [Point(np.array([-1, 0]), energy_table[starting_point[0] - 1, starting_point[1]]), Point(np.array([-1, -1]), energy_table[starting_point[0] - 1, starting_point[1] - 1]), Point(np.array([0, -1]), energy_table[starting_point[0], starting_point[1] - 1])]
        starting_point += (min(point_list, key=lambda point : point.value)).position
        carve_mask[starting_point[0], :starting_point[1]] = 255
    
    return carve_mask 

def fill_energy_table_diag(energy_table, image_gradient):
    energy_table[0] = image_gradient[0]
    for i in range(1, len(image_gradient)):
        for j in range(len(image_gradient[0])):
            if (j == 0): energy_table[i][j] = image_gradient[i][j] + min(energy_table[i - 1][j], energy_table[i - 1][j + 1])
            elif (j == len(image_gradient[0]) - 1): energy_table[i][j] = image_gradient[i][j] + min(energy_table[i - 1][j], energy_table[i - 1][j - 1], energy_table[i][j - 1], energy_table[i + 1][j - 1])
            else: energy_table[i][j] = image_gradient[i][j] + min(energy_table[i - 1][j], energy_table[i - 1][j - 1], energy_table[i][j - 1], energy_table[i - 1][j + 1], energy_table[i + 1][j - 1])

    return energy_table

def fill_energy_table(energy_table, image_gradient, horizontal = False):
    if (not horizontal):
        energy_table[0] = image_gradient[0] 
        for i in range(1, len(image_gradient)):
            for j in range(len(image_gradient[0])):
                if (j > 0 and j < len(image_gradient[0]) - 1): energy_table[i][j] = image_gradient[i][j] + min(energy_table[i - 1][j], energy_table[i - 1][j - 1], energy_table[i - 1][j + 1])
                elif (j == 0): energy_table[i][j] = image_gradient[i][j] + min(energy_table[i - 1][j], energy_table[i - 1][j + 1])
                else: energy_table[i][j] = image_gradient[i][j] + min(energy_table[i - 1][j], energy_table[i - 1][j - 1])
    else:
        energy_table[:, 0] = image_gradient[:, 0] 
        for i in range(1, len(image_gradient[0])): 
            for j in range(len(image_gradient)):
                if (j > 0 and j < (len(image_gradient) - 1)): energy_table[j][i] = image_gradient[j][i] + min(energy_table[j - 1][i - 1], energy_table[j][i - 1], energy_table[j + 1][i - 1])
                elif (j == 0): energy_table[j][i] = image_gradient[j][i] + min(energy_table[j][i - 1], energy_table[j + 1][i - 1])
                else: energy_table[j][i] = image_gradient[j][i] + min(energy_table[j - 1][i - 1], energy_table[j][i - 1])

    return energy_table

def traverse_energy_table(energy_table, horizontal=False):
    carve_mask = np.zeros(energy_table.shape).astype('uint8')
    current_sector, minval, subval = None, None, None
    compare_arr = None

    if (not horizontal):
        current_point = np.array([len(energy_table) - 1, np.argmin(energy_table[-1])]) # np.argmin(energy_table[-1]) instead of 0!
        while (current_point[0] > -1):
            carve_mask[current_point[0], :current_point[1]] = 255 # fills in the left 
            if (current_point[1] == 0): 
                compare_arr = energy_table[(current_point[0] - 1), current_point[1]:(current_point[1] + 2)]
                subval = 0
            elif (current_point[1] == (len(energy_table[0]) - 1)): 
                compare_arr = energy_table[(current_point[0] - 1), (current_point[1] - 1):(current_point[1] + 1)]
                subval = 1
            else: 
                compare_arr = energy_table[(current_point[0] - 1), (current_point[1] - 1):(current_point[1] + 2)]
                subval = 1
            current_point += (np.array([-1, np.argmin(compare_arr) - subval]))
    else:
        current_point = np.array([np.argmin(energy_table[:, -1]), len(energy_table[0]) - 1]) # np.argmin(energy_table[:, -1]) instead of 0
        while (current_point[1] > -1):
            carve_mask[current_point[0]:, current_point[1]] = 255 # fills in the bottom 
            if (current_point[0] == 0): 
                compare_arr = energy_table[(current_point[0]):(current_point[0] + 2), (current_point[1] - 1)]
                subval = 0
            elif (current_point[0] == (len(energy_table) - 1)): 
                compare_arr = energy_table[(current_point[0] - 1):(current_point[0] + 1), (current_point[1] - 1)]
                subval = 1
            else: 
                compare_arr = energy_table[(current_point[0] - 1):(current_point[0] + 2), (current_point[1] - 1)]
                subval = 1
            current_point += np.array([np.argmin(compare_arr) - subval, -1])
    
    return carve_mask

def determine_ending_edges(gradient_mask):
    top_edge = (get_edge(gradient_mask[0]))
    right_edge = (get_edge(gradient_mask[:, -1]))
    bottom_edge = (get_edge(gradient_mask[-1, :]))
    left_edge = (get_edge(gradient_mask[:, 0]))
    return top_edge, right_edge, bottom_edge, left_edge

def get_edge(image_line):
    start, end = -1, len(image_line)
    for i in range(len(image_line)):
        if image_line[i] > 0 and start == -1: start = i
        elif image_line[i] == 0 and start != -1: 
            end = i
            break
    return {'start': start, 'end': end, 'valid': True} if start != -1 else {'start': len(image_line), 'end': 0, 'valid': False}

def generate_image_gradient(error_image):
    error_image = cv2.GaussianBlur(error_image, (5, 5), 10)
    x_grad = cv2.Sobel(error_image, cv2.CV_64F, 1, 0, ksize=3)
    y_grad = cv2.Sobel(error_image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(np.abs(x_grad), 0.5, np.abs(y_grad), 0.5, 0).astype('uint64')

def get_error_between_overlap(input_image, base_image, overlap_mask):
    input_image = cv2.cvtColor(cv2.bitwise_and(input_image, input_image, mask=overlap_mask), cv2.COLOR_BGR2GRAY)
    base_image = cv2.cvtColor(cv2.bitwise_and(base_image, base_image, mask=overlap_mask), cv2.COLOR_BGR2GRAY)
    return np.square(cv2.addWeighted(base_image, -1, input_image, 1, 0))
    
# POTENTIAL - Construct homography matrix to determine if there are any rotations/transformations (other than translation) needed
# Determine where the new image will be placed - Allocate sectors for the present image and for the new image
# Blending - alpha or minimum error boundary to handle harsh edges then poisson to distribute lighting. 


# You can add to a buffer with slicing - it'll proabably make things a LOT easier 
# Plus you can do alpha blending a lot easier now! Yay! Also you could potentially utilize minimum error boundary cuts
# Also use poisson blending to handle variance in lighting 
# evidence supporting use of harris corner: https://cs.nyu.edu/~fergus/teaching/vision_2012/3_Corners_Blobs_Descriptors.pdf
# BRIEF image descriptor: https://medium.com/data-breach/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6 

# blending: https://web.stanford.edu/class/cs231m/lectures/lecture-5-stitching-blending.pdf