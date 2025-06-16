from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import numpy as np
import cv2
import os
import sys
import gc
from typing import List, Tuple, Optional
import time
import base64
import pickle
from io import BytesIO
import logging
import traceback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("spark_image_stitcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SparkImageStitcher:
    def __init__(self, spark_master="spark://10.0.42.43:7077"):
        """Initialize Spark session for image stitching"""
        logger.info("Initializing SparkImageStitcher...")
        
        try:
            self.spark = SparkSession.builder \
                .appName("DistributedImageStitching") \
                .master(spark_master) \
                .config("spark.local.dir", "C:\\spark-temp") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.ui.port", "4050") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.executor.memory", "4g") \
                .config("spark.executor.cores", "2") \
                .config("spark.driver.memory", "2g") \
                .config("spark.driver.maxResultSize", "1g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .getOrCreate()
            
            self.sc = self.spark.sparkContext
            logger.info(f"Spark session initialized successfully. Application ID: {self.sc.applicationId}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {str(e)}")
            raise

    def encode_image(self, img):
        """Encode image to base64 string for Spark serialization"""
        try:
            if img is None:
                logger.error("Cannot encode None image")
                return None
                
            _, buffer = cv2.imencode('.png', img)
            if not _:
                logger.error("Failed to encode image to PNG")
                return None
                
            img_str = base64.b64encode(buffer).decode()
            logger.debug(f"Image encoded successfully, size: {len(img_str)} characters")
            return img_str
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None

    def decode_image(self, img_str):
        """Decode base64 string back to image"""
        try:
            if not img_str:
                logger.error("Cannot decode empty image string")
                return None
                
            img_data = base64.b64decode(img_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image from base64")
                return None
                
            logger.debug(f"Image decoded successfully, shape: {img.shape}")
            return img
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return None

    def load_images_distributed(self, directory_path, supported_formats=None):
        """Load images in a distributed manner with enhanced error handling"""
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        supported_formats_lower = [ext.lower() for ext in supported_formats]
        logger.info(f"Searching for files with extensions: {supported_formats_lower}")

        logger.info(f"Loading images from: {directory_path}")

        try:
            # Validate directory
            if not os.path.exists(directory_path):
                logger.error(f"Directory does not exist: {directory_path}")
                return []
                
            if not os.path.isdir(directory_path):
                logger.error(f"Path is not a directory: {directory_path}")
                return []

            # Get list of image files
            all_files = os.listdir(directory_path)
            image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in supported_formats_lower]
            image_paths = [os.path.join(directory_path, f) for f in sorted(image_files)]

            logger.info(f"Found {len(image_paths)} image files")
            
            if len(image_paths) == 0:
                logger.warning("No image files found in directory")
                return []

            # Distribute image loading across cluster
            paths_rdd = self.sc.parallelize(image_paths)
            
            def load_and_encode_image(path):
                """Load image and encode for serialization with enhanced error handling"""
                import cv2
                import base64
                import numpy as np
                import os
                import logging
                logger = logging.getLogger("SparkImageStitcher.load_and_encode_image")
                try:
                    if not os.path.exists(path):
                        logger.warning(f"File does not exist: {path}")
                        return None
                    img = cv2.imread(path)
                    if img is None:
                        logger.warning(f"cv2.imread failed to load image: {path}")
                        return None
                    h, w = img.shape[:2]
                    max_width, max_height = 1200, 800
                    scale = min(max_width / w, max_height / h, 1.0)
                    if scale < 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    success, buffer = cv2.imencode('.png', img)
                    if not success:
                        logger.warning(f"cv2.imencode failed for image: {path}")
                        return None
                    img_str = base64.b64encode(buffer).decode()
                    return (os.path.basename(path), img_str, img.shape)
                except Exception as e:
                    logger.error(f"Exception loading image {path}: {e}")
                    return None

            # Load images in parallel
            loaded_images = paths_rdd.map(load_and_encode_image).filter(lambda x: x is not None).collect()
            
            logger.info(f"Successfully loaded {len(loaded_images)} out of {len(image_paths)} images")
            
            # Log details about loaded images
            for filename, _, shape in loaded_images:
                logger.debug(f"Loaded image: {filename}, shape: {shape}")
                
            logger.info(f'All files: {all_files}')
            logger.info(f'Image files: {image_files}')
            
            return loaded_images
            
        except Exception as e:
            logger.error(f"Error in load_images_distributed: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def extract_features_distributed(self, image_data_list):
        """Extract features from all images in parallel with enhanced logging"""
        logger.info(f"Extracting features from {len(image_data_list)} images...")
        
        if not image_data_list:
            logger.error("No images provided for feature extraction")
            return []
            
        detector_configs = {
            'SIFT': {'type': 'SIFT', 'nfeatures': 1000, 'contrastThreshold': 0.03, 'edgeThreshold': 15},
            'ORB': {'type': 'ORB', 'nfeatures': 1000, 'scaleFactor': 1.2, 'nlevels': 8},
            'AKAZE': {'type': 'AKAZE', 'threshold': 0.0003}
        }
        
        def extract_features_from_image(image_data):
            """Extract features from a single image with detailed logging"""
            import cv2
            import base64
            import numpy as np
            
            filename, img_str, shape = image_data
            
            try:
                # Decode image
                img_data = base64.b64decode(img_str)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return None
                
                features = {}
                
                for detector_name, config in detector_configs.items():
                    try:
                        # Create detector
                        if config['type'] == 'SIFT':
                            detector = cv2.SIFT_create(
                                nfeatures=config['nfeatures'],
                                contrastThreshold=config['contrastThreshold'],
                                edgeThreshold=config['edgeThreshold']
                            )
                        elif config['type'] == 'ORB':
                            detector = cv2.ORB_create(
                                nfeatures=config['nfeatures'],
                                scaleFactor=config['scaleFactor'],
                                nlevels=config['nlevels']
                            )
                        elif config['type'] == 'AKAZE':
                            detector = cv2.AKAZE_create(threshold=config['threshold'])
                        else:
                            continue
                        
                        # Extract features
                        kp, desc = detector.detectAndCompute(img, None)
                        
                        if desc is not None and len(kp) > 0:
                            # Convert keypoints to serializable format
                            kp_data = [p.pt for p in kp]
                            features[detector_name] = {
                                'keypoints': kp_data,
                                'descriptors': desc.tolist()
                            }
                        
                    except Exception as e:
                        continue
                
                return (filename, features, img_str, shape)
                
            except Exception as e:
                return None

        try:
            # Extract features in parallel
            images_rdd = self.sc.parallelize(image_data_list)
            features_rdd = images_rdd.map(extract_features_from_image)
            features_data = features_rdd.filter(lambda x: x is not None).collect()
            
            logger.info(f"Feature extraction completed for {len(features_data)} images")
            
            # Log feature extraction results
            for filename, features, _, _ in features_data:
                feature_counts = {detector: len(data['keypoints']) for detector, data in features.items()}
                logger.debug(f"Features extracted from {filename}: {feature_counts}")
            
            return features_data
            
        except Exception as e:
            logger.error(f"Error in extract_features_distributed: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def find_matches_parallel(self, features_data_list, match_ratio=0.75):
       
        logger.info(f"Finding matches between {len(features_data_list)} images with ratio {match_ratio}...")
        
        if len(features_data_list) < 2:
            logger.error("Need at least 2 images for matching")
            return []
        
        def match_image_pair(pair_data):
            
            import cv2
            import numpy as np
            
            idx1, idx2, feat1, feat2, img1_str, img2_str = pair_data
            filename1, features1, _, shape1 = feat1
            filename2, features2, _, shape2 = feat2
            
            try:
                best_matches = []
                best_score = 0
                best_detector = None
                
                # Try each detector type
                for detector_name in features1.keys():
                    if detector_name not in features2:
                        continue
                    
                    try:
                        # Get descriptors
                        desc1_data = features1[detector_name]['descriptors']
                        desc2_data = features2[detector_name]['descriptors']
                        
                        if not desc1_data or not desc2_data:
                            continue
                        
                        # Convert to numpy arrays with proper dtype
                        if detector_name == 'ORB':
                            desc1 = np.array(desc1_data, dtype=np.uint8)
                            desc2 = np.array(desc2_data, dtype=np.uint8)
                        else:
                            desc1 = np.array(desc1_data, dtype=np.float32)
                            desc2 = np.array(desc2_data, dtype=np.float32)
                        
                        # Validate descriptor shapes
                        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
                            continue
                            
                        if len(desc1.shape) != 2 or len(desc2.shape) != 2:
                            continue
                            
                        if desc1.shape[1] != desc2.shape[1]:
                            continue
                        
                        # Create matcher
                        if detector_name == 'ORB':
                            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                        else:
                            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                        
                        # K-nearest neighbors matching
                        matches = matcher.knnMatch(desc1, desc2, k=2)
                        
                        # Apply ratio test
                        good_matches = []
                        for match in matches:
                            if len(match) == 2:
                                m, n = match
                                if m.distance < match_ratio * n.distance:
                                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))
                        
                        # Calculate match quality score
                        if len(good_matches) > 0:
                            distances = [m[2] for m in good_matches]
                            avg_distance = np.mean(distances)
                            score = len(good_matches) / (1 + avg_distance)
                            
                            if score > best_score:
                                best_score = score
                                best_matches = good_matches
                                best_detector = detector_name
                    
                    except Exception as e:
                        continue
                
                return {
                    'pair': (idx1, idx2),
                    'filenames': (filename1, filename2),
                    'matches': best_matches,
                    'detector': best_detector,
                    'score': best_score,
                    'keypoints1': features1.get(best_detector, {}).get('keypoints', []) if best_detector else [],
                    'keypoints2': features2.get(best_detector, {}).get('keypoints', []) if best_detector else []
                }
                
            except Exception as e:
                return {
                    'pair': (idx1, idx2),
                    'filenames': (filename1, filename2),
                    'matches': [],
                    'detector': None,
                    'score': 0,
                    'keypoints1': [],
                    'keypoints2': [],
                    'error': str(e)
                }
        
        try:
            # Create pairs for matching
            pairs = []
            for i in range(len(features_data_list) - 1):
                pairs.append((i, i+1, features_data_list[i], features_data_list[i+1], 
                             features_data_list[i][2], features_data_list[i+1][2]))
            
            logger.info(f"Created {len(pairs)} image pairs for matching")
            
            # Match pairs in parallel
            pairs_rdd = self.sc.parallelize(pairs)
            matches_rdd = pairs_rdd.map(match_image_pair)
            match_results = matches_rdd.collect()
            
            # Log matching results
            for result in match_results:
                if 'error' in result:
                    logger.warning(f"Matching error for {result['filenames']}: {result['error']}")
                else:
                    logger.info(f"Matched {result['filenames']}: {len(result['matches'])} matches "
                              f"(detector: {result['detector']}, score: {result['score']:.3f})")
            
            return match_results
            
        except Exception as e:
            logger.error(f"Error in find_matches_parallel: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def warp_and_blend_all(self, images, homographies):
        logger.info("Starting global warping and simple feather blending of all images.")

        base_image_shape = images[0].shape
        h, w = base_image_shape[:2]

        corners_list = [np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)]
        for i in range(1, len(images)):
            h_i, w_i = images[i].shape[:2]
            corners_i = np.float32([[0, 0], [0, h_i], [w_i, h_i], [w_i, 0]]).reshape(-1, 1, 2)
            if homographies[i] is not None:
                corners_list.append(cv2.perspectiveTransform(corners_i, homographies[i]))

        all_corners = np.concatenate(corners_list, axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        canvas_width, canvas_height = x_max - x_min, y_max - y_min
        logger.info(f"Final canvas size: {canvas_width}x{canvas_height}")

        max_canvas_dim = 25000
        if canvas_width > max_canvas_dim or canvas_height > max_canvas_dim:
            logger.error(f"Canvas size ({canvas_width}x{canvas_height}) exceeds max dimension ({max_canvas_dim}).")
            return None

        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

        # Simple feather blending
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        weight_map = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)

        for i, img in enumerate(images):
            if homographies[i] is None:
                continue
            H_final = H_translation @ homographies[i]
            warped_img = cv2.warpPerspective(img.astype(np.float32), H_final, (canvas_width, canvas_height))
            mask = np.ones(img.shape[:2], dtype=np.float32)
            warped_mask = cv2.warpPerspective(mask, H_final, (canvas_width, canvas_height))
            warped_mask = np.expand_dims(warped_mask, axis=2)
            panorama += warped_img * warped_mask
            weight_map += warped_mask

        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        panorama = panorama / weight_map
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)

        # Crop as before
        logger.info("Cropping final panorama...")
        try:
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped = panorama[y:y+h, x:x+w]
                logger.info(f"Cropped to: {cropped.shape}")
                return cropped
            else:
                logger.warning("No contours found for cropping, returning full panorama.")
                return panorama
        except Exception as e:
            logger.warning(f"Cropping failed: {e}, returning uncropped image.")
            return panorama

    def stitch_center_referenced(self, features_data_list, min_match_count=10, match_ratio=0.6):
        """
        Stitch images by referencing all images to the center image (to reduce cumulative error).
        """
        logger.info(f"Starting center-referenced stitching with {len(features_data_list)} images...")
        if len(features_data_list) < 2:
            logger.error("Need at least 2 images for stitching")
            return None

        N = len(features_data_list)
        center_idx = N // 2
        logger.info(f"Using image at index {center_idx} as center reference: {features_data_list[center_idx][0]}")

        # Prepare homographies: identity for center, others to be computed
        homographies = [None] * N
        homographies[center_idx] = np.eye(3)

        # Helper to get image from features_data_list
        def decode_img(img_str):
            import base64, cv2, numpy as np
            img_data = base64.b64decode(img_str)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        center_img_str = features_data_list[center_idx][2]
        center_img = decode_img(center_img_str)
        center_features = features_data_list[center_idx][1]

        # Compute homographies to center
        for i, (filename, features, img_str, shape) in enumerate(features_data_list):
            if i == center_idx:
                continue
            img = decode_img(img_str)
            best_matches = []
            best_detector = None
            best_kp1 = best_kp2 = None
            # Try all detectors
            for detector_name in features.keys():
                if detector_name not in center_features:
                    continue
                kp1 = features[detector_name]['keypoints']
                kp2 = center_features[detector_name]['keypoints']
                desc1 = np.array(features[detector_name]['descriptors'], dtype=np.float32)
                desc2 = np.array(center_features[detector_name]['descriptors'], dtype=np.float32)
                if desc1.shape[0] == 0 or desc2.shape[0] == 0:
                    continue
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = matcher.knnMatch(desc1, desc2, k=2)
                good_matches = []
                for m in matches:
                    if len(m) == 2 and m[0].distance < match_ratio * m[1].distance:
                        good_matches.append(m[0])
                if len(good_matches) > len(best_matches):
                    best_matches = good_matches
                    best_detector = detector_name
                    best_kp1 = kp1
                    best_kp2 = kp2
            if len(best_matches) < min_match_count or best_kp1 is None or best_kp2 is None:
                logger.warning(f"Insufficient matches to center for image {filename} (found {len(best_matches)})")
                homographies[i] = None
                continue
            src_pts = np.float32([best_kp1[m.queryIdx] for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([best_kp2[m.trainIdx] for m in best_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                homographies[i] = H
                logger.info(f"Computed homography to center for image {filename}")
            else:
                logger.warning(f"Failed to compute homography to center for image {filename}")
                homographies[i] = None

        # Decode all images
        images = [decode_img(f[2]) for f in features_data_list]
        # Blend all images using their homographies
        panorama = self.warp_and_blend_all(images, homographies)
        return panorama

    def stitch_images_distributed(self, directory_path, min_match_count=10, match_ratio=0.6):
        """Main distributed stitching pipeline with comprehensive logging"""
        
        logger.info("=" * 70)
        logger.info("DISTRIBUTED IMAGE STITCHING WITH SPARK")
        logger.info("=" * 70)
        logger.info(f"Input directory: {directory_path}")
        logger.info(f"Minimum match count: {min_match_count}")
        logger.info(f"Match ratio threshold: {match_ratio}")
        logger.info("=" * 70)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Load images in parallel
            logger.info("\n" + "="*50)
            logger.info("STEP 1: LOADING IMAGES")
            logger.info("="*50)
            start_time = time.time()
            image_data_list = self.load_images_distributed(directory_path)
            load_time = time.time() - start_time
            
            logger.info(f"Image loading completed in {load_time:.2f} seconds")
            logger.info(f"Images loaded: {len(image_data_list)}")
            
            if len(image_data_list) < 2:
                logger.error(f"Insufficient images for stitching (found {len(image_data_list)}, need ≥2)")
                return None
            
            # Step 2: Extract features in parallel
            logger.info("\n" + "="*50)
            logger.info("STEP 2: FEATURE EXTRACTION")
            logger.info("="*50)
            start_time = time.time()
            features_data_list = self.extract_features_distributed(image_data_list)
            feature_time = time.time() - start_time
            
            logger.info(f"Feature extraction completed in {feature_time:.2f} seconds")
            logger.info(f"Images with features: {len(features_data_list)}")
            
            if len(features_data_list) < 2:
                logger.error(f"Insufficient images with features (found {len(features_data_list)}, need ≥2)")
                return None
            
            # Step 3: Find matches in parallel
            logger.info("\n" + "="*50)
            logger.info("STEP 3: FEATURE MATCHING")
            logger.info("="*50)
            start_time = time.time()
            match_results = self.find_matches_parallel(features_data_list, match_ratio)
            match_time = time.time() - start_time
            
            logger.info(f"Feature matching completed in {match_time:.2f} seconds")
            logger.info(f"Image pairs processed: {len(match_results)}")
            
            # Display detailed match results
            logger.info("\nMATCH RESULTS SUMMARY:")
            logger.info("-" * 50)
            total_matches = 0
            valid_pairs = 0
            
            for i, result in enumerate(match_results):
                match_count = len(result['matches'])
                total_matches += match_count
                status = "✓ VALID" if match_count >= min_match_count else "✗ INSUFFICIENT"
                if match_count >= min_match_count:
                    valid_pairs += 1
                
                logger.info(f"Pair {i+1}: {result['filenames'][0]} → {result['filenames'][1]}")
                logger.info(f"  Matches: {match_count} | Detector: {result['detector']} | "
                          f"Score: {result['score']:.3f} | {status}")
            
            logger.info(f"\nTotal matches found: {total_matches}")
            logger.info(f"Valid pairs (≥{min_match_count} matches): {valid_pairs}/{len(match_results)}")
            
            if valid_pairs == 0:
                logger.error("No valid image pairs found for stitching!")
                return None
            
            # Step 4: Center-referenced stitching
            logger.info("\n" + "="*50)
            logger.info("STEP 4: CENTER-REFERENCED STITCHING")
            logger.info("="*50)
            start_time = time.time()
            final_panorama = self.stitch_center_referenced(features_data_list, min_match_count, match_ratio)
            stitch_time = time.time() - start_time
            logger.info(f"Center-referenced stitching completed in {stitch_time:.2f} seconds")
            
            # Calculate and display timing summary
            total_time = time.time() - total_start_time
            
            logger.info("\n" + "="*70)
            logger.info("TIMING SUMMARY")
            logger.info("="*70)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"  • Image loading:     {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
            logger.info(f"  • Feature extraction: {feature_time:.2f}s ({feature_time/total_time*100:.1f}%)")
            logger.info(f"  • Feature matching:   {match_time:.2f}s ({match_time/total_time*100:.1f}%)")
            logger.info(f"  • Image stitching:    {stitch_time:.2f}s ({stitch_time/total_time*100:.1f}%)")
            logger.info("="*70)
            
            if final_panorama is not None:
                logger.info(f"SUCCESS! Final panorama shape: {final_panorama.shape}")
                return final_panorama
            else:
                logger.error("FAILED! Could not create final panorama")
                return None
                
        except Exception as e:
            logger.error(f"Fatal error in stitch_images_distributed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def stop(self):
        """Stop Spark session with logging"""
        try:
            logger.info("Stopping Spark session...")
            self.spark.stop()
            logger.info("Spark session stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Spark session: {str(e)}")

def main():
    """Main execution with enhanced error handling and logging"""
    
    logger.info("="*70)
    logger.info("SPARK DISTRIBUTED IMAGE STITCHING")
    logger.info("="*70)
    
    # Initialize Spark Image Stitcher
    stitcher = None
    
    try:
        logger.info("Initializing Spark Image Stitcher...")
        stitcher = SparkImageStitcher(spark_master="spark://10.0.42.43:7077")
        
        # Configuration
        custom_dir = r'C:\Users\user\Desktop\Keshav\Nesac_Outreach_Mapping'
        output_filename = "spark_distributed_panorama.jpg"
        
        logger.info(f"Input directory: {custom_dir}")
        logger.info(f"Output filename: {output_filename}")
        
        # Validate input directory
        if not os.path.exists(custom_dir):
            logger.error(f"Input directory does not exist: {custom_dir}")
            return
        
        # Perform distributed stitching
        logger.info("Starting distributed stitching process...")
        final_panorama = stitcher.stitch_images_distributed(
            directory_path=custom_dir,
            min_match_count=10,
            match_ratio=0.6
        )
        
        # Handle results
        if final_panorama is not None:
            try:
                # Save result
                logger.info(f"Saving panorama to: {output_filename}")
                success = cv2.imwrite(output_filename, final_panorama)
                
                if success:
                    logger.info(f"✓ Panorama saved successfully: {output_filename}")
                    
                    # Log file info
                    file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
                    logger.info(f"  File size: {file_size:.2f} MB")
                    logger.info(f"  Image dimensions: {final_panorama.shape}")
                else:
                    logger.error("Failed to save panorama image")
                
                # Optional: Display result (if running locally with GUI)
                try:
                    import matplotlib.pyplot as plt
                    logger.info("Attempting to display result...")
                    
                    plt.figure(figsize=(20, 12))
                    plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
                    plt.axis("off")
                    plt.title("SPARK DISTRIBUTED PANORAMA", fontsize=18, fontweight='bold', pad=20)
                    plt.tight_layout()
                    plt.savefig("panorama_preview.png", dpi=150, bbox_inches='tight')
                    plt.show()
                    
                    logger.info("✓ Panorama displayed and preview saved")
                    
                except ImportError:
                    logger.info("Matplotlib not available - skipping display")
                except Exception as e:
                    logger.warning(f"Could not display panorama: {e}")
                    
            except Exception as e:
                logger.error(f"Error saving/displaying panorama: {str(e)}")
                
        else:
            logger.error("Could not create panorama with distributed processing")
            logger.info("Possible issues:")
            logger.info("  • Insufficient matching features between images")
            logger.info("  • Images may not have sufficient overlap")
            logger.info("  • Try adjusting match_ratio or min_match_count parameters")
            logger.info("  • Check image quality and lighting conditions")
    
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        # Always stop Spark session
        if stitcher is not None:
            try:
                stitcher.stop()
            except Exception as e:
                logger.error(f"Error stopping stitcher: {str(e)}")
        
        logger.info("Program execution completed")

if __name__ == "__main__":
    main()