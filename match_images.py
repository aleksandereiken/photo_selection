import other_functions
import cv2

def image_matching(img_path_1, img_path_2, cutoff=300, n_matches=20, show_matches=False):
    img1 = cv2.imread(img_path_1, 0)
    img2 = cv2.imread(img_path_2, 0)
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate sum of distane for first nfeatures matches
    sum = 0
    for match in matches[:n_matches]:
        sum += match.distance

    if show_matches:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_matches], None)
        other_functions.show_image(match_img)

    if sum < cutoff:
        print(f"\tMatches with sum of distance sum of {sum}")
        return True
    else:
        print(f"\tNo matches with sum of distance sum of {sum}")
        return False

def image_matching_from_PIL_img(img1, img2, cutoff=300, n_matches=20, show_matches=False):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate sum of distane for first nfeatures matches
    sum = 0
    for match in matches[:n_matches]:
        sum += match.distance

    if show_matches:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_matches], None)
        other_functions.show_image(match_img)

    if sum < cutoff:
        print(f"Matches with sum of distance sum of {sum}")
        return True
    else:
        print(f"No matches with sum of distance sum of {sum}")
        return False

def match_images_in_groups(current_group_df, n_pictures_in_group):
    # Subgroup based on whether the images match
    for index_group in range(n_pictures_in_group - 1):
        # Get file names
        if index_group == 0:
            comp_img = current_group_df["file_names"][0]
            # Create lists for holding groups and counter
            match_groups = [0]
            counter = 0

        next_file = current_group_df["file_names"][index_group + 1]

        # Group images based on matches
        match = image_matching(comp_img, next_file, cutoff= 700, n_matches= 20, show_matches= False)
        if match:
            match_groups.append(counter)
        else:
            # Assign new group
            counter += 1

            # Overwrite comp_image with next_file
            comp_img = next_file

            match_groups.append(counter)

    current_group_df["subgroups"] = match_groups
    return current_group_df