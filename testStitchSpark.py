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
from collections import deque, defaultdict


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
        """Initialize Spark session with optimized configuration"""
        logger.info("Initializing SparkImageStitcher...")
        
        try:
            # Get the current Python executable path
            import sys
            python_executable = sys.executable
            logger.info(f"Using Python executable: {python_executable}")
            
            # Configure Spark with optimized settings
            conf = SparkConf()
            conf.setMaster(spark_master)
            conf.setAppName("SparkImageStitcher")
            
            # Memory and performance settings
            conf.set("spark.driver.memory", "10g")
            conf.set("spark.executor.memory", "12g")
            conf.set("spark.driver.maxResultSize", "8g")
            conf.set("spark.sql.adaptive.enabled", "true")
            conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
            
            # Serialization settings
            conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            conf.set("spark.kryoserializer.buffer.max", "2047m")
            conf.set("spark.kryoserializer.buffer", "1024m")
            
            # Python version compatibility - ensure same Python version on all nodes
            conf.set("spark.pyspark.python", python_executable)
            conf.set("spark.pyspark.driver.python", python_executable)
            conf.set("spark.python.worker.python", python_executable)
            
            # Set environment variables for Python version consistency
            import os
            os.environ['PYSPARK_PYTHON'] = python_executable
            os.environ['PYSPARK_DRIVER_PYTHON'] = python_executable
            
            # Network and timeout settings
            conf.set("spark.network.timeout", "800s")
            conf.set("spark.executor.heartbeatInterval", "60s")
            conf.set("spark.dynamicAllocation.enabled", "false")
            
            # Logging level
            conf.set("spark.eventLog.enabled", "false")
            
            # Create Spark session
            self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
            self.sc = self.spark.sparkContext
            
            # Set log level
            self.sc.setLogLevel("WARN")
            
            logger.info(f"Spark session initialized successfully. Application ID: {self.sc.applicationId}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {str(e)}")
            logger.error(traceback.format_exc())
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
                    
                    # More aggressive resolution reduction for large datasets
                    # Start with smaller max dimensions for better memory management
                    max_width, max_height = 600, 450  # Reduced from 800x600
                    
                    # If we have many images, reduce resolution further
                    if len(image_paths) > 30:
                        max_width, max_height = 400, 300
                    elif len(image_paths) > 20:
                        max_width, max_height = 500, 375
                    
                    scale = min(max_width / w, max_height / h, 1.0)
                    if scale < 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        logger.debug(f"Resized {os.path.basename(path)} from {w}x{h} to {new_w}x{new_h}")
                    
                    success, buffer = cv2.imencode('.png', img)
                    if not success:
                        logger.warning(f"cv2.imencode failed for image: {path}")
                        return None
                    img_str = base64.b64encode(buffer).decode()
                    
                    # Force garbage collection
                    gc.collect()
                    
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
            'SIFT': {'type': 'SIFT', 'nfeatures': 500, 'contrastThreshold': 0.04, 'edgeThreshold': 10},
            'ORB': {'type': 'ORB', 'nfeatures': 500, 'scaleFactor': 1.2, 'nlevels': 8},
            'AKAZE': {'type': 'AKAZE', 'threshold': 0.0008}
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
            finally:
                # Force garbage collection
                gc.collect()

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

    def find_matches_parallel(self, features_data_list, match_ratio=0.75, adjacent_only=True):
       
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
                        desc1_data = features1[detector_name]['descriptors']
                        desc2_data = features2[detector_name]['descriptors']
                        if not desc1_data or not desc2_data:
                            continue
                        if detector_name == 'ORB':
                            desc1 = np.array(desc1_data, dtype=np.uint8)
                            desc2 = np.array(desc2_data, dtype=np.uint8)
                        else:
                            desc1 = np.array(desc1_data, dtype=np.float32)
                            desc2 = np.array(desc2_data, dtype=np.float32)
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
                        # KNN match for Lowe's ratio test
                        matches_1to2 = matcher.knnMatch(desc1, desc2, k=2)
                        good_matches_1to2 = []
                        for match in matches_1to2:
                            if len(match) == 2:
                                m, n = match
                                if m.distance < match_ratio * n.distance:
                                    good_matches_1to2.append((m.queryIdx, m.trainIdx, m.distance))
                        # Cross-check: match in the other direction
                        matches_2to1 = matcher.knnMatch(desc2, desc1, k=2)
                        good_matches_2to1 = []
                        for match in matches_2to1:
                            if len(match) == 2:
                                m, n = match
                                if m.distance < match_ratio * n.distance:
                                    good_matches_2to1.append((m.queryIdx, m.trainIdx, m.distance))
                        # Cross-check filter: only keep matches that agree in both directions
                        cross_checked_matches = []
                        set_2to1 = set((m[1], m[0]) for m in good_matches_2to1)
                        for m in good_matches_1to2:
                            if (m[0], m[1]) in set_2to1:
                                cross_checked_matches.append(m)
                        # Calculate match quality score
                        if len(cross_checked_matches) > 0:
                            distances = [m[2] for m in cross_checked_matches]
                            avg_distance = np.mean(distances)
                            score = len(cross_checked_matches) / (1 + avg_distance)
                            if score > best_score:
                                best_score = score
                                best_matches = cross_checked_matches
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
            if adjacent_only:
                # Only match adjacent images (for sequential stitching)
                for i in range(len(features_data_list) - 1):
                    pairs.append((i, i+1, features_data_list[i], features_data_list[i+1], 
                                 features_data_list[i][2], features_data_list[i+1][2]))
                logger.info(f"Created {len(pairs)} adjacent image pairs for matching")
            else:
                # Match all possible pairs (for graph-based stitching)
                for i in range(len(features_data_list)):
                    for j in range(i+1, len(features_data_list)):
                        pairs.append((i, j, features_data_list[i], features_data_list[j], 
                                     features_data_list[i][2], features_data_list[j][2]))
                logger.info(f"Created {len(pairs)} total image pairs for matching")
            
            # Process matches in batches to reduce memory usage
            batch_size = 10  # Process 10 pairs at a time
            all_match_results = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(pairs) + batch_size - 1)//batch_size} ({len(batch_pairs)} pairs)")
                
                # Match pairs in parallel for this batch
                pairs_rdd = self.sc.parallelize(batch_pairs)
                matches_rdd = pairs_rdd.map(match_image_pair)
                batch_results = matches_rdd.collect()
                
                # Log matching results for this batch
                for result in batch_results:
                    if 'error' in result:
                        logger.warning(f"Matching error for {result['filenames']}: {result['error']}")
                    else:
                        logger.info(f"Matched {result['filenames']}: {len(result['matches'])} matches "
                                  f"(detector: {result['detector']}, score: {result['score']:.3f})")
                
                all_match_results.extend(batch_results)
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
            
            return all_match_results
            
        except Exception as e:
            logger.error(f"Error in find_matches_parallel: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def warp_and_blend_all(self, images, homographies):
        """Warp and blend all images using global homographies"""
        logger.info("Starting global warping and simple feather blending of all images.")
        
        if not images or not homographies:
            logger.error("No images or homographies provided")
            return None
        
        # Calculate canvas bounds
        all_corners = []
        for i, img in enumerate(images):
            if homographies[i] is not None:
                h, w = img.shape[:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                warped_corners = cv2.perspectiveTransform(corners, homographies[i])
                all_corners.append(warped_corners.reshape(-1, 2))
        
        if not all_corners:
            logger.error("No valid homographies found")
            return None
        
        all_corners = np.vstack(all_corners)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        canvas_width, canvas_height = x_max - x_min, y_max - y_min
        logger.info(f"Calculated canvas size: {canvas_width}x{canvas_height}")

        # Check if canvas is too large and apply scaling if needed
        max_canvas_dim = 20000  # Reduced from 25000 for safety
        scale_factor = 1.0
        
        if canvas_width > max_canvas_dim or canvas_height > max_canvas_dim:
            scale_factor = min(max_canvas_dim / canvas_width, max_canvas_dim / canvas_height)
            canvas_width = int(canvas_width * scale_factor)
            canvas_height = int(canvas_height * scale_factor)
            logger.info(f"Canvas too large, applying scale factor: {scale_factor:.3f}")
            logger.info(f"Scaled canvas size: {canvas_width}x{canvas_height}")
            
            # Apply scaling to homographies
            scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=np.float64)
            homographies = [h @ scale_matrix if h is not None else None for h in homographies]
            
            # Recalculate corners with scaled homographies
            all_corners = []
            for i, img in enumerate(images):
                if homographies[i] is not None:
                    h, w = img.shape[:2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    warped_corners = cv2.perspectiveTransform(corners, homographies[i])
                    all_corners.append(warped_corners.reshape(-1, 2))
            
            all_corners = np.vstack(all_corners)
            x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            canvas_width, canvas_height = x_max - x_min, y_max - y_min
            logger.info(f"Final scaled canvas size: {canvas_width}x{canvas_height}")

        # Safety check for extremely large canvases
        if canvas_width > 25000 or canvas_height > 25000:
            logger.error(f"Canvas size ({canvas_width}x{canvas_height}) still exceeds maximum allowed dimension (25000).")
            logger.error("Consider reducing the number of images or using a smaller dataset.")
            return None

        # Create the translation matrix to shift images to the positive quadrant
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

        # Check memory requirements before creating canvas
        estimated_memory_mb = (canvas_width * canvas_height * 3 * 4) / (1024 * 1024)  # 4 bytes per float32
        logger.info(f"Estimated memory requirement: {estimated_memory_mb:.1f} MB")
        
        if estimated_memory_mb > 8000:  # 8GB limit
            logger.error(f"Estimated memory requirement ({estimated_memory_mb:.1f} MB) exceeds 8GB limit.")
            logger.error("Consider reducing image resolution or number of images.")
            return None

        try:
            panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_map = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)
        except MemoryError:
            logger.error("Memory error while creating canvas. Canvas too large.")
            return None

        logger.info(f"Processing {len(images)} images for blending...")
        
        for i, img in enumerate(images):
            if homographies[i] is not None:
                try:
                    H_final = H_translation @ homographies[i]
                    warped_img = cv2.warpPerspective(img.astype(np.float32), H_final, (canvas_width, canvas_height))
                    
                    # Create a mask for blending
                    mask = np.ones(img.shape[:2], dtype=np.float32)
                    warped_mask = cv2.warpPerspective(mask, H_final, (canvas_width, canvas_height))
                    warped_mask = np.expand_dims(warped_mask, axis=2)
                    
                    panorama += warped_img * warped_mask
                    weight_map += warped_mask
                    
                    logger.debug(f"Processed image {i+1}/{len(images)}")
                except Exception as e:
                    logger.warning(f"Error processing image {i}: {e}")
                    continue

        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        panorama = (panorama / weight_map).clip(0, 255).astype(np.uint8)

        # Save uncropped version
        try:
            cv2.imwrite("spark_distributed_panorama_uncropped.jpg", panorama)
            logger.info("Saved uncropped panorama")
        except Exception as e:
            logger.warning(f"Could not save uncropped panorama: {e}")

        # Crop the panorama
        logger.info("Cropping final panorama...")
        try:
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cropped = panorama[y:y+h, x:x+w]
                logger.info(f"Cropped panorama size: {cropped.shape[1]}x{cropped.shape[0]}")
                return cropped
            else:
                logger.warning("No contours found for cropping.")
                return panorama
        except Exception as e:
            logger.warning(f"Cropping failed: {e}, returning uncropped image.")
            return panorama

    def stitch_sequentially(self, match_results, features_data_list, min_match_count):
        """Sequential stitching method with fallback for large datasets"""
        logger.info("Starting sequential stitching...")
        if not match_results:
            logger.error("No match results for sequential stitching")
            return None

        images = [self.decode_image(f[2]) for f in features_data_list]
        
        # Try with all images first
        panorama = self._try_stitch_with_images(images, match_results, features_data_list, min_match_count)
        
        if panorama is not None:
            return panorama
        
        # If failed, try with fewer images (every 2nd image)
        logger.warning("Stitching failed with all images. Trying with reduced dataset...")
        reduced_images = images[::2]  # Take every 2nd image
        reduced_features = features_data_list[::2]
        
        # Create reduced match results for the subset
        reduced_matches = []
        for i in range(len(reduced_images) - 1):
            # Find the corresponding match result for this pair
            for result in match_results:
                if (result['pair'][0] == i*2 and result['pair'][1] == i*2 + 1):
                    # Adjust pair indices for reduced dataset
                    adjusted_result = result.copy()
                    adjusted_result['pair'] = (i, i+1)
                    reduced_matches.append(adjusted_result)
                    break
        
        if len(reduced_matches) > 0:
            panorama = self._try_stitch_with_images(reduced_images, reduced_matches, reduced_features, min_match_count)
            if panorama is not None:
                logger.info("Successfully created panorama with reduced dataset")
                return panorama
        
        # If still failed, try with even fewer images (every 3rd image)
        logger.warning("Stitching failed with reduced dataset. Trying with minimal dataset...")
        minimal_images = images[::3]  # Take every 3rd image
        minimal_features = features_data_list[::3]
        
        minimal_matches = []
        for i in range(len(minimal_images) - 1):
            for result in match_results:
                if (result['pair'][0] == i*3 and result['pair'][1] == i*3 + 1):
                    adjusted_result = result.copy()
                    adjusted_result['pair'] = (i, i+1)
                    minimal_matches.append(adjusted_result)
                    break
        
        if len(minimal_matches) > 0:
            panorama = self._try_stitch_with_images(minimal_images, minimal_matches, minimal_features, min_match_count)
            if panorama is not None:
                logger.info("Successfully created panorama with minimal dataset")
                return panorama
        
        logger.error("All stitching attempts failed")
        return None

    def _try_stitch_with_images(self, images, match_results, features_data_list, min_match_count):
        """Helper function to attempt stitching with a given set of images"""
        logger.info(f"Attempting to stitch {len(images)} images...")
        
        # Calculate pairwise homographies (H_i+1 -> H_i)
        pairwise_homographies = []
        for result in match_results:
            if result['matches'] and len(result['matches']) >= min_match_count:
                detector_name = result['detector']
                if detector_name:
                    try:
                        kp1_data = features_data_list[result['pair'][0]][1][detector_name]['keypoints']
                        kp2_data = features_data_list[result['pair'][1]][1][detector_name]['keypoints']
                        
                        src_pts = np.float32([kp2_data[m[1]] for m in result['matches'] if m[1] < len(kp2_data)])
                        dst_pts = np.float32([kp1_data[m[0]] for m in result['matches'] if m[0] < len(kp1_data)])

                        if len(src_pts) >= 4:
                            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            pairwise_homographies.append(H)
                        else:
                            pairwise_homographies.append(None)
                    except Exception as e:
                        logger.error(f"Error computing homography for {result['filenames']}: {e}")
                        pairwise_homographies.append(None)
                else:
                    pairwise_homographies.append(None)
            else:
                pairwise_homographies.append(None)

        # Convert pairwise to global homographies (H_i -> H_0)
        logger.info("Calculating cumulative homographies...")
        global_homographies = [np.identity(3)]
        h_cumulative = np.identity(3)
        for h_pairwise in pairwise_homographies:
            if h_pairwise is not None:
                h_cumulative = h_cumulative @ h_pairwise
                global_homographies.append(h_cumulative)
            else:
                logger.warning("Broken link in homography chain. Stitching will stop here.")
                remaining = len(images) - len(global_homographies)
                global_homographies.extend([None] * remaining)
                break

        panorama = self.warp_and_blend_all(images, global_homographies)
        
        if panorama is None:
            logger.error("Panorama creation failed in warp_and_blend_all.")
            return None

        valid_homographies = [h for h in global_homographies if h is not None]
        logger.info(f"Computed {len(valid_homographies)} valid global homographies.")
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
            
            # Force garbage collection after image loading
            import gc
            gc.collect()
            
            if len(image_data_list) < 2:
                logger.error(f"Insufficient images for stitching (found {len(image_data_list)}, need >=2)")
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
            
            # Force garbage collection after feature extraction
            gc.collect()
            
            if len(features_data_list) < 2:
                logger.error(f"Insufficient images with features (found {len(features_data_list)}, need >=2)")
                return None
            
            # Step 3: Find matches in parallel
            logger.info("\n" + "="*50)
            logger.info("STEP 3: FEATURE MATCHING")
            logger.info("="*50)
            start_time = time.time()
            match_results = self.find_matches_parallel(features_data_list, match_ratio, adjacent_only=True)
            match_time = time.time() - start_time
            
            logger.info(f"Feature matching completed in {match_time:.2f} seconds")
            logger.info(f"Image pairs processed: {len(match_results)}")
            
            # Force garbage collection after feature matching
            gc.collect()
            
            # Display detailed match results
            logger.info("\nMATCH RESULTS SUMMARY:")
            logger.info("-" * 50)
            total_matches = 0
            valid_pairs = 0
            
            for i, result in enumerate(match_results):
                match_count = len(result['matches'])
                total_matches += match_count
                status = "VALID" if match_count >= min_match_count else "X INSUFFICIENT"
                if match_count >= min_match_count:
                    valid_pairs += 1
                
                logger.info(f"Pair {i+1}: {result['filenames'][0]} -> {result['filenames'][1]}")
                logger.info(f"  Matches: {match_count} | Detector: {result['detector']} | "
                          f"Score: {result['score']:.3f} | {status}")
            
            logger.info(f"\nTotal matches found: {total_matches}")
            logger.info(f"Valid pairs (>={min_match_count} matches): {valid_pairs}/{len(match_results)}")
            
            if valid_pairs == 0:
                logger.error("No valid image pairs found for stitching!")
                return None
            
            # Step 4: Sequential stitching
            logger.info("\n" + "="*50)
            logger.info("STEP 4: SEQUENTIAL STITCHING")
            logger.info("="*50)
            start_time = time.time()
            final_panorama = self.stitch_sequentially(match_results, features_data_list, min_match_count)
            stitch_time = time.time() - start_time
            logger.info(f"Sequential stitching completed in {stitch_time:.2f} seconds")
            
            # Calculate and display timing summary
            total_time = time.time() - total_start_time
            
            logger.info("\n" + "="*70)
            logger.info("TIMING SUMMARY")
            logger.info("="*70)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"  * Image loading:     {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
            logger.info(f"  * Feature extraction: {feature_time:.2f}s ({feature_time/total_time*100:.1f}%)")
            logger.info(f"  * Feature matching:   {match_time:.2f}s ({match_time/total_time*100:.1f}%)")
            logger.info(f"  * Image stitching:    {stitch_time:.2f}s ({stitch_time/total_time*100:.1f}%)")
            logger.info("="*70)
            
            if final_panorama is not None:
                logger.info(f"SUCCESS! Final panorama shape: {final_panorama.shape}")
                # --- Post-processing: Sharpening and Contrast Enhancement ---
                try:
                    import cv2
                    import numpy as np
                    # Sharpening
                    gaussian = cv2.GaussianBlur(final_panorama, (0, 0), 3)
                    sharpened = cv2.addWeighted(final_panorama, 1.5, gaussian, -0.5, 0)
                    # CLAHE Contrast Enhancement
                    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    # Save both versions
                    cv2.imwrite("spark_distributed_panorama_sharpened.jpg", sharpened)
                    cv2.imwrite("spark_distributed_panorama_enhanced.jpg", enhanced)
                    logger.info("Sharpened and contrast-enhanced panoramas saved as 'spark_distributed_panorama_sharpened.jpg' and 'spark_distributed_panorama_enhanced.jpg'")
                except Exception as e:
                    logger.error(f"Post-processing failed: {e}")
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
        custom_dir = r'C:\Users\user\Desktop\Keshav\Shillong_Fire_MAPPING'
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
                    logger.info(f" Panorama saved successfully: {output_filename}")
                    
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
                    
                    logger.info("Panorama displayed and preview saved")
                    
                except ImportError:
                    logger.info("Matplotlib not available - skipping display")
                except Exception as e:
                    logger.warning(f"Could not display panorama: {e}")
                    
            except Exception as e:
                logger.error(f"Error saving/displaying panorama: {str(e)}")
                
        else:
            logger.error("Could not create panorama with distributed processing")
            logger.info("Possible issues:")
            logger.info("  * Insufficient matching features between images")
            logger.info("  * Images may not have sufficient overlap")
            logger.info("  * Try adjusting match_ratio or min_match_count parameters")
            logger.info("  * Check image quality and lighting conditions")
    
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