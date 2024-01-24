import cv2
import numpy as np
import ants


def main():
    src_path = r'/home/kobi/PycharmProjects/SuperPointPretrainedNetwork/assets/data/source.png'
    dest_path = r'/home/kobi/PycharmProjects/SuperPointPretrainedNetwork/assets/data/cropped_dest.png'

    src_image = cv2.imread(src_path)
    dest_image = cv2.imread(dest_path)

    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    dest_image = cv2.cvtColor(dest_image, cv2.COLOR_BGR2RGB)

    cv_registration(src_image, dest_image)
    # ants_registration(src_image, dest_image)


def ants_registration(src_image, dest_image):
    gray_src = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    gray_dest = cv2.cvtColor(dest_image, cv2.COLOR_RGB2GRAY)

    moving_image = ants.from_numpy(gray_src.T)
    fixed_image = ants.from_numpy(gray_dest.T)

    mytx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Elastic', verbose=False)

    # fixed_image.plot(overlay=mytx['warpedmovout'], title='After registration')

    result = cv2.addWeighted(fixed_image.numpy().T, 0.7, mytx['warpedmovout'].numpy().T, 0.3, 0)
    collage = np.concatenate([cv2.resize(gray_src, (300, 300)),
                                    cv2.resize(gray_dest, (300, 300)),
                                    cv2.resize(result, (300, 300))], axis=1)

    window_size = (1600, 1000)
    cv2.namedWindow('After registration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('After registration', *window_size)
    cv2.imshow('After registration', collage)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def cv_registration(src_image, dest_image):
    # Initialize ORB detector
    # orb = cv2.ORB_create()
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(src_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(dest_image, None)

    matching_method = flan
    matching_method(src_image, dest_image, keypoints1, keypoints2, descriptors1, descriptors2)

    exit()

    image1 = dest_image
    image2 = src_image

    # Filter out good matches
    good_matches = [m for m in matches if m.distance < 55]

    # Get corresponding points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get the dimensions of image2
    height, width = image2.shape[:2]

    # Define the four corners of image2
    corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], dtype=np.float32).reshape(-1,
                                                                                                                     1,
                                                                                                                     2)

    # Warp the corners of image2 to image1 using the homography matrix
    warped_corners = cv2.perspectiveTransform(corners, homography_matrix)

    # Calculate the bounding box for the warped image2 in image1
    min_x, min_y = np.int32(warped_corners.min(axis=0).ravel())
    max_x, max_y = np.int32(warped_corners.max(axis=0).ravel())

    # Calculate the size of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Create a mask for the region of interest (ROI)
    mask = np.zeros_like(image1, dtype=np.uint8)
    cv2.fillPoly(mask, [warped_corners.astype(np.int32)], 255)

    # Extract the region of interest in image1
    roi = image1[min_y:max_y, min_x:max_x]

    # Warp image2 to fit the region in image1
    registered_image = cv2.warpPerspective(image2, homography_matrix, (width, height))

    # Blend the warped image2 with the ROI in image1
    result = cv2.addWeighted(roi, 0.5, registered_image, 0.5, 0)

    # Replace the ROI in image1 with the blended result
    image1[min_y:max_y, min_x:max_x] = result

    # Display the result
    cv2.imshow('Registered Image', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bf(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2):
    # Create a BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches on the images
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matches


def flan(image1, image2, keypoints1, keypoints2, descriptors1, descriptors2):
    # FLANN parameters
    flann_params = dict(algorithm=6,  # FLANN_INDEX_LSH is recommended for binary descriptors like ORB
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(flann_params, {})

    # Match descriptors using KNN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    window_size = (1600, 1000)
    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Matches', *window_size)

    # Display the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matches


if __name__ == "__main__":
    main()
