import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def draw_points(image, points):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw key points
    for kp in points:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    return image


def find_interest_points(image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect key points and descriptors
    key_points, descriptors = sift.detectAndCompute(image, None)

    # Draw key points
    image_with_key_points = draw_points(image, key_points)
    return image_with_key_points, key_points, descriptors


def find_matching_points(descriptors1, descriptors2):
    # Initialize the Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches_ = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches_ = sorted(matches_, key=lambda x: x.distance)
    return matches_


def draw_matches_separately(image1_, key_points1, image2_, key_points2, matches_):
    image1_with_matches = image1_.copy()
    image2_with_matches = image2_.copy()

    image1_with_matches = cv2.cvtColor(image1_with_matches, cv2.COLOR_GRAY2BGR)
    image2_with_matches = cv2.cvtColor(image2_with_matches, cv2.COLOR_GRAY2BGR)

    # Draw the matching points on each image separately
    for match in matches_:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        x1, y1 = int(key_points1[img1_idx].pt[0]), int(key_points1[img1_idx].pt[1])
        x2, y2 = int(key_points2[img2_idx].pt[0]), int(key_points2[img2_idx].pt[1])

        cv2.circle(image1_with_matches, (x1, y1), 2, (0, 0, 255), -1)
        cv2.circle(image2_with_matches, (x2, y2), 2, (0, 0, 255), -1)

    return image1_with_matches, image2_with_matches


def draw_matches_stitched(image1_, key_points1, image2_, key_points2, matches_):
    image1_with_matches = image1_.copy()
    image2_with_matches = image2_.copy()

    image1_with_matches = cv2.cvtColor(image1_with_matches, cv2.COLOR_GRAY2BGR)
    image2_with_matches = cv2.cvtColor(image2_with_matches, cv2.COLOR_GRAY2BGR)

    # Find max dimensions for the stitched image
    max_height = max(image1_with_matches.shape[0], image2_with_matches.shape[0])
    total_width = image1_with_matches.shape[1] + image2_with_matches.shape[1]

    stitched_image_ = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    stitched_image_[:image1_with_matches.shape[0], :image1_with_matches.shape[1], :] = image1_with_matches
    stitched_image_[:image2_with_matches.shape[0], image1_with_matches.shape[1]:, :] = image2_with_matches

    # Draw the matching points on stitched_image with colored circles and lines
    for match_ in matches_[:70]:  # Taking a subset of 70 matches
        img1_idx = match_.queryIdx
        img2_idx = match_.trainIdx

        x1, y1 = int(key_points1[img1_idx].pt[0]), int(key_points1[img1_idx].pt[1])
        x2, y2 = int(key_points2[img2_idx].pt[0]) + image1_with_matches.shape[1], int(key_points2[img2_idx].pt[1])

        # Draw circles and lines
        cv2.circle(stitched_image_, (x1, y1), 5, (0, 0, 255), 1)  # Red circle for image 1
        cv2.circle(stitched_image_, (x2, y2), 5, (0, 255, 0), 1)  # Green circle for image 2
        cv2.line(stitched_image_, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow line between matches

    return stitched_image_


def compute_essential_matrix(matches_, key_points1, key_points2, K_matrix):
    # Extract matched points
    points1 = np.array([key_points1[m.queryIdx].pt for m in matches_])
    points2 = np.array([key_points2[m.trainIdx].pt for m in matches_])

    # Compute essential matrix
    E_matrix, mask = cv2.findEssentialMat(points1, points2, K_matrix, method=cv2.RANSAC, prob=0.999, threshold=0.1)
    return E_matrix, mask.ravel().astype(bool)


def compute_fundamental_matrix(E_matrix, K_matrix):
    # Compute Fundamental Matrix F from Essential Matrix E and Intrinsic Matrix K
    K_inv = np.linalg.inv(K_matrix)
    F_matrix = np.dot(np.dot(K_inv.T, E_matrix), K_inv)
    return F_matrix


def draw_epipolar_line(image, line, color):
    rows, cols = image.shape[:2]
    a, b, c = line.ravel()
    x0, y0 = 0, int(-c / b)
    x1, y1 = cols, int(-(c + a * cols) / b)
    bgr_color = tuple(np.array(color[:3]) * 255)
    cv2.line(image, (x0, y0), (x1, y1), bgr_color, 1)


def get_epipolar_lines_images(image1_, key_points1, image2_, key_points2, matches_, F_matrix):
    epipolar_lines_image_1 = image1_.copy()
    epipolar_lines_image_2 = image2_.copy()

    epipolar_lines_image_1 = cv2.cvtColor(epipolar_lines_image_1, cv2.COLOR_GRAY2BGR)
    epipolar_lines_image_2 = cv2.cvtColor(epipolar_lines_image_2, cv2.COLOR_GRAY2BGR)

    # Generate a colormap with distinct colors
    colormap = plt.colormaps['hsv']

    # Draw the matching points on each image separately
    for i, match in enumerate(matches_):
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        x1, y1 = int(key_points1[img1_idx].pt[0]), int(key_points1[img1_idx].pt[1])
        x2, y2 = int(key_points2[img2_idx].pt[0]), int(key_points2[img2_idx].pt[1])

        # Draw the match points
        cv2.circle(epipolar_lines_image_1, (x1, y1), 5, (0, 255, 0), 1)
        cv2.circle(epipolar_lines_image_2, (x2, y2), 5, (0, 255, 0), 1)

        color = colormap(i)

        # Compute the epipolar lines in both images
        point1 = np.array([x1, y1, 1]).reshape(3, 1)
        point2 = np.array([x2, y2, 1]).reshape(3, 1)

        epipolar_line1 = F_matrix.T @ point2  # Epipolar line in image1 corresponding to point2
        epipolar_line2 = F_matrix @ point1  # Epipolar line in image2 corresponding to point1

        draw_epipolar_line(epipolar_lines_image_1, epipolar_line1, color)
        draw_epipolar_line(epipolar_lines_image_2, epipolar_line2, color)

    return epipolar_lines_image_1, epipolar_lines_image_2


def compute_camera_poses(E_matrix, K_matrix):
    # Perform SVD on the Essential matrix
    U, S, Vt = np.linalg.svd(E_matrix)

    # Define the matrix W used to ensure proper rotation matrices
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))

    # Translation vector
    t = U[:, 2]

    # Two possible translations
    t1, t2 = t, -t

    # Camera matrix for the first camera
    P1_ = np.dot(K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Generate four possible camera matrices for the second camera
    P2_options_ = [
        np.dot(K_matrix, np.hstack((R1, t1.reshape(3, 1)))),
        np.dot(K_matrix, np.hstack((R1, t2.reshape(3, 1)))),
        np.dot(K_matrix, np.hstack((R2, t1.reshape(3, 1)))),
        np.dot(K_matrix, np.hstack((R2, t2.reshape(3, 1))))
    ]

    return P1_, P2_options_


def triangulate_points(P1_, P2_, key_points1, key_points2, matches_):
    # Extract 2D points from the matched key points in both images
    points1 = np.array([key_points1[m.queryIdx].pt for m in matches_]).T
    points2 = np.array([key_points2[m.trainIdx].pt for m in matches_]).T

    # Perform triangulation to obtain 4D homogeneous coordinates
    points_4d_hom = cv2.triangulatePoints(P1_, P2_, points1, points2)

    # Convert 4D homogeneous coordinates to 3D by dividing by the last row
    points_3d_ = points_4d_hom / points_4d_hom[3]
    return points_3d_[:3].T


def fit_planes_ransac(points, distance_threshold):
    planes_ = []
    remaining_points = points.copy()

    while len(remaining_points) >= 10:
        # Fit a plane using RANSAC
        model = LinearRegression()
        ransac = RANSACRegressor(model, residual_threshold=distance_threshold, random_state=42)
        ransac.fit(remaining_points[:, :2], remaining_points[:, 2])

        # Get inliers and their mask
        inlier_mask = ransac.inlier_mask_
        inliers = remaining_points[inlier_mask]

        if len(inliers) < 3:
            break

        # Fit the plane parameters
        plane_normal = np.array([ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], -1])
        plane_point = np.mean(inliers, axis=0)

        planes_.append((plane_normal, plane_point))

        # Remove inliers from the remaining points
        remaining_points = remaining_points[~inlier_mask]

    return planes_


def color_points_by_plane(points, planes_, distance_threshold):
    # Define specific colors for the planes
    colors = np.array([[0, 0, 1],    # Blue
                       [1, 0, 0],    # Red
                       [0, 1, 0],    # Green
                       [1, 1, 0],    # Yellow
                       [0, 1, 1]])   # Cyan

    # Initialize point colors to white
    point_colors_ = np.ones((points.shape[0], 3))

    # Assign colors to each point based on the plane it belongs to
    for plane_idx, (normal, point) in enumerate(planes_):
        # Compute distances from the point to the plane
        distances = np.abs(np.dot(points - point, normal)) / np.linalg.norm(normal)

        # Check which points are close enough to the plane using the distance threshold
        inliers = distances < distance_threshold
        point_colors_[inliers] = colors[plane_idx % len(colors)]

    return point_colors_


def project_points_to_2d(points_3d_, P):
    # Projects 3D points to 2D using the camera projection matrix P
    points_3d_hom = np.hstack((points_3d_, np.ones((points_3d_.shape[0], 1))))
    points_2d_hom = points_3d_hom @ P.T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]
    return points_2d


def draw_colored_points_on_image(image, points_2d, colors):
    # Draws colored points on the image
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y), color in zip(points_2d, colors):
        if not np.all(color == [1.0, 1.0, 1.0]):
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            cv2.circle(image_colored, (int(x), int(y)), 2, color_bgr, 5)
    return image_colored


def calculate_normals_for_points(points, planes_):
    normals = np.zeros_like(points)

    # Assign normal vectors to each point based on the plane it belongs to
    for normal, point in planes_:
        # Compute distances from the point to the plane
        distances = np.abs(np.dot(points - point, normal)) / np.linalg.norm(normal)

        # Check which points are close enough to the plane
        inliers = distances < 0.5
        normals[inliers] = normal / np.linalg.norm(normal)  # Normalize the normal vector

    return normals


def project_normals_to_2d(points_3d_, normals_3d_, P, scale=1):
    # Project 3D points to 2D
    points_2d = project_points_to_2d(points_3d_, P)

    # Project the end points of the normals
    normal_end_points_3d = points_3d_ + normals_3d_ * scale
    normal_end_points_2d = project_points_to_2d(normal_end_points_3d, P)

    return points_2d, normal_end_points_2d


def draw_normals_on_image(image, points_2d, normal_end_points_2d, colors):
    image_with_normals = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the normals on the image
    for (x1, y1), (x2, y2), color in zip(points_2d, normal_end_points_2d, colors):
        if not np.all(color == [1.0, 1.0, 1.0]):
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            cv2.line(image_with_normals, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)

    return image_with_normals


if __name__ == "__main__":

    # load the images
    path1 = "data/example_3/I1.png"
    path2 = "data/example_3/I2.png"

    image1 = load_image(path1)
    image2 = load_image(path2)

    # ------------------------------- Sec. 1 -------------------------------
    # Find the interest point in each image
    interest_points_img1, key_points_1, descriptors_1 = find_interest_points(image1)
    interest_points_img2, key_points_2, descriptors_2 = find_interest_points(image2)

    interest_points_img1 = cv2.cvtColor(interest_points_img1, cv2.COLOR_BGR2RGB)
    interest_points_img2 = cv2.cvtColor(interest_points_img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(interest_points_img1)
    plt.title("Interest Points - Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(interest_points_img2)
    plt.title("Interest Points -  Image 2")

    plt.show()

    # ------------------------------- Sec. 2 -------------------------------
    # Find matches between the descriptors
    matches = find_matching_points(descriptors_1, descriptors_2)

    # Draw matches separately
    matches_image_1, matches_image_2 = draw_matches_separately(image1, key_points_1, image2, key_points_2, matches)

    matches_image_1 = cv2.cvtColor(matches_image_1, cv2.COLOR_BGR2RGB)
    matches_image_2 = cv2.cvtColor(matches_image_2, cv2.COLOR_BGR2RGB)

    # Create stitched image with matches
    stitched_image = draw_matches_stitched(image1, key_points_1, image2, key_points_2, matches)
    stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

    # Display the matches images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(matches_image_1)
    plt.title("Matches - Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(matches_image_2)
    plt.title("Matches - Image 2")

    # Display the stitched image
    plt.figure(figsize=(10, 5))
    plt.imshow(stitched_image_rgb)
    plt.title("Image with Matching Subset")
    plt.axis('off')
    plt.show()

    # ------------------------------- Sec. 3 -------------------------------
    # ********* a *********
    # Extract K matrix
    file_path = 'data/example_3/K.txt'
    K = np.loadtxt(file_path, delimiter=',')

    # Compute essential matrix E and inlier mask
    E, mask_E = compute_essential_matrix(matches, key_points_1, key_points_2, K)
    np.set_printoptions(suppress=True, formatter={'float': '{:0.15f}'.format})
    print("Essential Matrix:")
    print(E)

    # Compute Fundamental Matrix F
    F = compute_fundamental_matrix(E, K)
    np.set_printoptions(suppress=True, formatter={'float': '{:0.15f}'.format})
    print("\nFundamental Matrix:")
    print(F)

    # ********* b *********
    filtered_matches = [matches[i] for i, m in enumerate(mask_E) if m]

    # Create stitched image with matches
    stitched_image_E_filtered = draw_matches_stitched(image1, key_points_1, image2, key_points_2, filtered_matches)
    stitched_image_E_filtered_rgb = cv2.cvtColor(stitched_image_E_filtered, cv2.COLOR_BGR2RGB)

    # Display the stitched image
    plt.figure(figsize=(10, 5))
    plt.imshow(stitched_image_E_filtered_rgb)
    plt.title("Image with Filtered Matches")
    plt.axis('off')
    plt.show()

    # ********* c *********
    # Choose 50 random matches from the inlier matches
    num_samples = 50
    random_filtered_matches = random.sample(filtered_matches, min(num_samples, len(filtered_matches)))

    # Plot the Epiplar lines on each image
    epipolar_1, epipolar_2 = get_epipolar_lines_images(image1, key_points_1,
                                                       image2, key_points_2, random_filtered_matches, F)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image 1 - Epipolar Lines")
    plt.imshow(epipolar_1)

    plt.subplot(1, 2, 2)
    plt.title("Image 2 - Epipolar Lines")
    plt.imshow(epipolar_2)
    plt.show()

    # ------------------------------- Sec.4 -------------------------------
    # ********* a *********
    # Get the camera poses matrices - P1 and the options for P2
    P1, P2_options = compute_camera_poses(E, K)

    # ********* b *********
    points_3d = []
    best_P2 = []
    max_positive_depth_count = 0

    # Find the correct camera pose
    for idx, P2 in enumerate(P2_options):
        points_3d = triangulate_points(P1, P2, key_points_1, key_points_2, filtered_matches)

        # Count the number of points with positive z-coordinates
        positive_depth_count = np.sum(points_3d[:, 2] > 0)

        # Update the best pose if the current one has more positive depths
        if positive_depth_count > max_positive_depth_count:
            max_positive_depth_count = positive_depth_count
            best_P2 = P2

    P2 = best_P2

    # Visualize the 3d points cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    ax.set_xlim(-5, 15)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 40)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # ------------------------------- Sec.5 -------------------------------
    # ********* a *********
    # Find the planes by the 3d points cloud using RANSAC
    planes = fit_planes_ransac(points_3d, distance_threshold=1.5)
    point_colors = color_points_by_plane(points_3d, planes, distance_threshold=0.5)

    # Visualize the 3d points cloud colored by planes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=point_colors, marker='o')
    ax.set_xlim(-5, 15)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 40)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title("Colored 3D Point Cloud by Plane")
    plt.show()

    # ********* b *********
    # Project 3D points to 2D on both images
    points_2d_image1 = project_points_to_2d(points_3d, np.array(P1))
    points_2d_image2 = project_points_to_2d(points_3d, np.array(P2))

    # Draw colored points on the images
    image1_colored = draw_colored_points_on_image(image1, points_2d_image1, point_colors)
    image2_colored = draw_colored_points_on_image(image2, points_2d_image2, point_colors)

    # Convert images from BGR to RGB for Matplotlib
    image1_colored_rgb = cv2.cvtColor(image1_colored, cv2.COLOR_BGR2RGB)
    image2_colored_rgb = cv2.cvtColor(image2_colored, cv2.COLOR_BGR2RGB)

    # Display the images with points colored by plane
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1_colored_rgb)
    plt.title("Image 1 with Colored Points")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2_colored_rgb)
    plt.title("Image 2 with Colored Points")
    plt.axis('off')

    plt.show()

    # ------------------------------- Sec.6 -------------------------------

    # Calculate normal vectors for points
    normals_3d = calculate_normals_for_points(points_3d, planes)

    # Project normal vectors to 2D
    points_2d_image1, normal_end_points_2d_image1 = project_normals_to_2d(points_3d, normals_3d, np.array(P1))
    points_2d_image2, normal_end_points_2d_image2 = project_normals_to_2d(points_3d, normals_3d, np.array(P2))

    image1_with_normals = draw_normals_on_image(image1, points_2d_image1, normal_end_points_2d_image1, point_colors)
    image2_with_normals = draw_normals_on_image(image2, points_2d_image2, normal_end_points_2d_image2, point_colors)

    image1_with_normals_rgb = cv2.cvtColor(image1_with_normals, cv2.COLOR_BGR2RGB)
    image2_with_normals_rgb = cv2.cvtColor(image2_with_normals, cv2.COLOR_BGR2RGB)

    # Display the images with normals
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1_with_normals_rgb)
    plt.title("Image 1 with Normals")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2_with_normals_rgb)
    plt.title("Image 2 with Normals")
    plt.axis('off')

    plt.show()
