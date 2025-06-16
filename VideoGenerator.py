#Import library
#import libraries
import sys
import numpy as np

#folfer containing images from drones, sorted by name
import glob
import cv2
import os

import matplotlib.pyplot as plt
import os
import sys
import gc
from typing import List, Tuple, Optional
import time

# Video writer for saving intermediary results
class VideoLogger:
    def __init__(self, output_path="stitching_process.mp4", fps=2, frame_duration=3):
        self.output_path = output_path
        self.fps = fps
        self.frame_duration = frame_duration  # seconds to show each frame
        self.writer = None
        self.frame_size = None
        self.frames = []
        
    def add_frame(self, img, title="", resize_to=(1920, 1080)):
        """Add a frame to the video with optional title"""
        if img is None:
            return
            
        # Resize frame to consistent size
        frame = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        
        # Add title overlay if provided
        if title:
            # Create a copy to avoid modifying original
            frame = frame.copy()
            
            # Add semi-transparent overlay for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (len(title) * 20 + 20, 60), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Add text
            cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Store frame (repeat for duration)
        for _ in range(self.fps * self.frame_duration):
            self.frames.append(frame)
    
    def add_side_by_side_frame(self, img1, img2, title1="Image 1", title2="Image 2", main_title=""):
        """Add side by side comparison frame"""
        if img1 is None or img2 is None:
            return
            
        # Resize images to same height
        target_height = 800
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        
        resized_img1 = cv2.resize(img1, (new_w1, target_height))
        resized_img2 = cv2.resize(img2, (new_w2, target_height))
        
        # Create side by side image
        total_width = new_w1 + new_w2 + 20  # 20px gap
        combined = np.zeros((target_height, total_width, 3), dtype=np.uint8)
        combined[:, :new_w1] = resized_img1
        combined[:, new_w1+20:] = resized_img2
        
        # Add individual titles
        cv2.putText(combined, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, title2, (new_w1 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        self.add_frame(combined, main_title, resize_to=(1920, 1080))
    
    def add_matches_frame(self, img1, img2, kp1, kp2, matches, title="Feature Matches"):
        """Add feature matches visualization frame"""
        if len(matches) > 0:
            # Limit matches for better visualization
            display_matches = matches[:min(50, len(matches))]
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, display_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            full_title = f"{title} ({len(matches)} total matches)"
            self.add_frame(img_matches, full_title)
    
    def save_video(self):
        """Save all frames as video"""
        if not self.frames:
            print("No frames to save")
            return
            
        print(f"Saving video with {len(self.frames)} frames to {self.output_path}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = self.frames[0].shape[:2][::-1]  # (width, height)
        
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, frame_size)
        
        # Write all frames
        for frame in self.frames:
            self.writer.write(frame)
        
        self.writer.release()
        print(f"✅ Video saved: {self.output_path}")

def show_image(img, title="Image", figsize=(12, 8)):
    """Display image with minimal text - now just passes to video logger"""
    pass  # We'll handle this in the video logger

def show_side_by_side(img1, img2, title1="Image 1", title2="Image 2"):
    """Show two images side by side - now just passes to video logger"""
    pass  # We'll handle this in the video logger

def show_matches(img1, img2, kp1, kp2, good_matches, title="Feature Matches"):
    """Visualize feature matches between two images - now just passes to video logger"""
    pass  # We'll handle this in the video logger

def resize_image(img, max_width=1200, max_height=800):
    """Resize image if too large"""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img

def load_images_from_directory(directory_path, video_logger=None, supported_formats=None):
    """Load images from directory with video logging"""
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    print(f"Loading images from: {directory_path}")

    original_dir = os.getcwd()
    try:
        os.chdir(directory_path)
    except Exception as e:
        print(f"Error: {e}")
        return [], [], {}

    # Find image files
    all_files = os.listdir()
    image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in supported_formats)]
    path = sorted(set(image_files))

    print(f"Found {len(path)} images")

    # Load images
    img_list = []
    img_paths = []

    for img_path in path:
        img = cv2.imread(img_path)
        if img is not None:
            img = resize_image(img)
            img_list.append(img)
            img_paths.append(img_path)

    # Show loaded images in video
    if len(img_list) > 0 and video_logger:
        print(f"Successfully loaded {len(img_list)} images")

        # Add loaded images to video
        for i, (img, path) in enumerate(zip(img_list[:4], img_paths[:4])):
            video_logger.add_frame(img, f"Loaded Image {i+1}: {path}")

    os.chdir(original_dir)
    return img_list, img_paths, {}

def geometric_verification(kp1, kp2, matches, max_reproj_error=5.0):
    """Apply geometric verification to filter out bad matches"""
    if len(matches) < 4:
        return matches, None

    # Extract point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Try multiple homography estimation methods
    methods = [
        (cv2.RANSAC, "RANSAC"),
        (cv2.LMEDS, "LMEDS"),
        (cv2.RHO, "RHO")
    ]

    best_inlier_count = 0
    best_mask = None
    best_homography = None

    for method, method_name in methods:
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, method, max_reproj_error)
            if H is not None and mask is not None:
                inlier_count = np.sum(mask)
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_mask = mask
                    best_homography = H
        except:
            continue

    if best_mask is not None:
        # Filter matches based on inliers
        verified_matches = [matches[i] for i in range(len(matches)) if best_mask[i]]
        return verified_matches, best_homography
    else:
        return matches, None

def spatial_consistency_check(kp1, kp2, matches, neighbor_distance=50):
    """Check spatial consistency of matches"""
    if len(matches) < 10:
        return matches

    # Convert keypoints to arrays
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

    consistent_matches = []

    for i, match in enumerate(matches):
        # Find neighbors in first image
        distances = np.linalg.norm(pts1 - pts1[i], axis=1)
        neighbors_idx = np.where((distances < neighbor_distance) & (distances > 0))[0]

        if len(neighbors_idx) < 2:
            consistent_matches.append(match)
            continue

        # Check if relative positions are consistent in second image
        consistent_count = 0
        for j in neighbors_idx:
            # Relative position in image 1
            rel_pos1 = pts1[j] - pts1[i]
            # Relative position in image 2
            rel_pos2 = pts2[j] - pts2[i]

            # Check if relative positions are similar
            if np.linalg.norm(rel_pos1 - rel_pos2) < neighbor_distance * 0.5:
                consistent_count += 1

        # Keep match if majority of neighbors are consistent
        if consistent_count >= len(neighbors_idx) * 0.3:
            consistent_matches.append(match)

    return consistent_matches

def warpImages(img1, img2, H, upscale_factor=1.0):
    """Warp and blend two images using homography matrix"""
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Get corner points of both images
    corners_img1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # Warp corners of img2
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)

    # Bounding box
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-x_min, -y_min]

    # Translation homography
    H_translation = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    # Warp img2
    warped_img2 = cv2.warpPerspective(img2, H_translation @ H, (x_max - x_min, y_max - y_min), flags=cv2.INTER_LINEAR)

    # Place img1 on top
    warped_img2[translation[1]:translation[1]+rows1, translation[0]:translation[0]+cols1] = img1

    # Crop out large black areas
    gray = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped = warped_img2[y:y+h, x:x+w]
    else:
        cropped = warped_img2

    final = cropped
    return final

def create_multiple_detectors():
    """Create multiple feature detectors for robust matching"""
    detectors = {}

    # SIFT - good for general purpose
    detectors['SIFT'] = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, edgeThreshold=15)

    # ORB - faster, good for textured scenes
    detectors['ORB'] = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

    # AKAZE - good for natural scenes
    detectors['AKAZE'] = cv2.AKAZE_create(threshold=0.0003)

    return detectors

def find_matches_robust(img1, img2, match_ratio=0.75, use_multiple_detectors=True):
    """Enhanced feature matching with multiple strategies"""

    detectors = create_multiple_detectors()
    best_matches = []
    best_kp1, best_kp2 = None, None
    best_score = 0
    best_detector = None

    for detector_name, detector in detectors.items():
        try:
            # Detect keypoints and descriptors
            if detector_name == 'ORB':
                kp1, desc1 = detector.detectAndCompute(img1, None)
                kp2, desc2 = detector.detectAndCompute(img2, None)
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                norm_type = cv2.NORM_HAMMING
            else:
                kp1, desc1 = detector.detectAndCompute(img1, None)
                kp2, desc2 = detector.detectAndCompute(img2, None)
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                norm_type = cv2.NORM_L2

            if desc1 is None or desc2 is None:
                continue

            # K-nearest neighbors matching
            matches = matcher.knnMatch(desc1, desc2, k=2)

            # Apply ratio test
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < match_ratio * n.distance:
                        good_matches.append(m)

            # Calculate match quality score
            if len(good_matches) > 0:
                distances = [m.distance for m in good_matches]
                avg_distance = np.mean(distances)
                score = len(good_matches) / (1 + avg_distance)  # More matches + lower distance = better

                print(f"{detector_name}: {len(good_matches)} matches, score: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_matches = good_matches
                    best_kp1, best_kp2 = kp1, kp2
                    best_detector = detector_name

        except Exception as e:
            print(f"Error with {detector_name}: {e}")
            continue

    print(f"Best detector: {best_detector} with {len(best_matches)} matches")
    return best_matches, len(best_matches), best_score, best_kp1, best_kp2

def stitch_images_robust(img_list: List[np.ndarray],
                        img_paths: List[str] = None,
                        video_logger: VideoLogger = None,
                        min_match_count=8,
                        match_ratio=0.6,
                        upscale_factor=1.5,
                        show_matches_viz=True,
                        use_geometric_verification=True,
                        use_spatial_consistency=True):
    """
    Robust image stitching with advanced feature matching and video logging
    """

    if len(img_list) < 2:
        print("Need at least 2 images")
        return None

    img_queue = [img.copy() for img in img_list]
    path_queue = img_paths.copy() if img_paths else [f"image_{i}" for i in range(len(img_list))]

    stitch_count = 1
    current_result = None

    print(f"\nStarting robust stitching with {len(img_queue)} images...")
    print(f"Using geometric verification: {use_geometric_verification}")
    print(f"Using spatial consistency: {use_spatial_consistency}")

    while len(img_queue) > 1:
        img1 = img_queue.pop(0)
        img2 = img_queue.pop(0)
        path1 = path_queue.pop(0) if path_queue else f"result_{stitch_count-1}"
        path2 = path_queue.pop(0) if path_queue else f"image_{stitch_count}"

        print(f"\n--- Step {stitch_count}: Stitching {path1} + {path2} ---")

        # Add input images to video
        if video_logger:
            video_logger.add_side_by_side_frame(img1, img2, f"Input 1: {path1}", f"Input 2: {path2}", 
                                              f"Step {stitch_count}: Input Images")

        # Find matches using robust method
        good_matches, total_matches, match_quality, kp1, kp2 = find_matches_robust(
            img1, img2, match_ratio
        )

        print(f"Initial matches: {len(good_matches)}")

        # Apply spatial consistency check
        if use_spatial_consistency and len(good_matches) > 10:
            good_matches = spatial_consistency_check(kp1, kp2, good_matches)
            print(f"After spatial consistency: {len(good_matches)}")

        # Apply geometric verification
        homography = None
        if use_geometric_verification and len(good_matches) >= 4:
            good_matches, homography = geometric_verification(kp1, kp2, good_matches)
            print(f"After geometric verification: {len(good_matches)}")

        # Add matches visualization to video
        if show_matches_viz and len(good_matches) > 0 and video_logger:
            video_logger.add_matches_frame(img1, img2, kp1, kp2, good_matches, 
                                         f"Step {stitch_count} - Verified Matches")

        if len(good_matches) >= min_match_count and kp1 is not None and kp2 is not None:
            try:
                # Use pre-computed homography or compute new one
                if homography is None:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    homography, mask = cv2.findHomography(src_pts, dst_pts,
                                                        cv2.RANSAC, 5.0, maxIters=5000)

                if homography is not None:
                    # Additional homography validation
                    det = np.linalg.det(homography[:2, :2])
                    if 0.1 < abs(det) < 10:  # Reasonable scale change
                        # Perform stitching
                        result = warpImages(img2, img1, homography, upscale_factor=1.0)
                        current_result = result
                        img_queue.insert(0, result)
                        path_queue.insert(0, f"stitched_step_{stitch_count}")

                        print(f"✅ Success! Homography determinant: {det:.3f}")

                        # Add result to video
                        if video_logger:
                            video_logger.add_frame(result, f"Step {stitch_count} - Robust Stitched Result")
                    else:
                        raise Exception(f"Invalid homography (det={det:.3f})")
                else:
                    raise Exception("Homography computation failed")

            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                # Fallback strategy
                if current_result is not None:
                    img_queue.insert(0, current_result)
                    path_queue.insert(0, f"previous_result")
                else:
                    better_img = img1 if img1.shape[0] * img1.shape[1] > img2.shape[0] * img2.shape[1] else img2
                    better_path = path1 if img1.shape[0] * img1.shape[1] > img2.shape[0] * img2.shape[1] else path2
                    img_queue.insert(0, better_img)
                    path_queue.insert(0, better_path)
        else:
            print(f"❌ Insufficient verified matches (need {min_match_count})")
            # Fallback strategy
            if current_result is not None:
                img_queue.insert(0, current_result)
                path_queue.insert(0, f"previous_result")
            else:
                better_img = img1 if img1.shape[0] * img1.shape[1] > img2.shape[0] * img2.shape[1] else img2
                better_path = path1 if img1.shape[0] * img1.shape[1] > img2.shape[0] * img2.shape[1] else path2
                img_queue.insert(0, better_img)
                path_queue.insert(0, better_path)

        stitch_count += 1
        gc.collect()

    # Final result
    if len(img_queue) == 1:
        final_result = img_queue[0]

        # Apply final upscaling
        if upscale_factor > 1.0:
            print(f"Applying final upscaling ({upscale_factor}x)")
            final_result = cv2.resize(final_result, None,
                                    fx=upscale_factor, fy=upscale_factor,
                                    interpolation=cv2.INTER_CUBIC)

        # Add final result to video
        if video_logger:
            video_logger.add_frame(final_result, "🏆 FINAL ROBUST PANORAMA")

        print(f"\n🎉 ROBUST STITCHING COMPLETE! Final size: {final_result.shape}")
        return final_result
    else:
        print("❌ Robust stitching failed")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution with video logging"""

    # Set your directory path here
    custom_dir = r'C:\Users\user\Desktop\Keshav\Nesac_Outreach_Mapping'  # <-- Change this path

    print("🚀 Robust Image Stitching with Video Output")
    print("=" * 50)

    # Initialize video logger
    video_logger = VideoLogger(
        output_path="stitching_process.mp4",
        fps=1,  # 1 frame per second for slower viewing
        frame_duration=4  # Show each step for 4 seconds
    )

    # Load images
    img_list, img_paths, _ = load_images_from_directory(custom_dir, video_logger)

    if len(img_list) < 2:
        print(f"❌ Need at least 2 images (found {len(img_list)})")
        return

    # Perform robust stitching
    final_panorama = stitch_images_robust(
        img_list=img_list,
        img_paths=img_paths,
        video_logger=video_logger,
        min_match_count=10,
        match_ratio=0.6,
        upscale_factor=1.5,
        show_matches_viz=True,
        use_geometric_verification=True,
        use_spatial_consistency=True
    )

    # Save final result as image
    if final_panorama is not None:
        cv2.imwrite("robust_panorama.jpg", final_panorama)
        print("💾 Saved final panorama: robust_panorama.jpg")
    else:
        print("❌ Could not create panorama with robust matching")

    # Save the video
    video_logger.save_video()
    print(f"🎬 Video saved! You can now watch the stitching process in: {video_logger.output_path}")

# Run the main function
if __name__ == "__main__":
    main()