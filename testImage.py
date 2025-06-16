from pyspark.sql import SparkSession
import numpy as np
import cv2
import os
import base64
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("image_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DistributedImageProcessor:
    def __init__(self, spark_master="spark://10.0.42.43:7077"):
        logger.info("Initializing Spark Session...")
        self.spark = SparkSession.builder \
            .appName("DistributedImageProcessing") \
            .master(spark_master) \
            .config("spark.local.dir", "C:\\spark-temp") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.maxResultSize", "1g") \
            .getOrCreate()
        self.sc = self.spark.sparkContext
        logger.info("Spark Session initialized.")

    def process_images(self, input_dir, output_dir):
        logger.info(f"Processing images from: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # List all image paths
        all_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ]

        image_rdd = self.sc.parallelize(all_files)

        logger.info(f"Number of files found: {len(all_files)}")

        def process_image(path):
            import cv2
            import numpy as np
            import os
            import base64

            try:
                filename = os.path.basename(path)
                with open(path, "rb") as f:
                    content = f.read()

                img_array = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is None:
                    return None

                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                success, buffer = cv2.imencode(".jpg", processed)

                if not success:
                    return None

                encoded = base64.b64encode(buffer).decode("utf-8")
                return (filename, encoded)

            except Exception as e:
                return None

        processed_images = image_rdd.map(process_image).filter(lambda x: x is not None).collect()

        for filename, base64_str in processed_images:
            try:
                out_path = os.path.join(output_dir, filename)
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(base64_str))
                logger.info(f"Saved processed image: {out_path}")
            except Exception as e:
                logger.error(f"Failed to save image {filename}: {str(e)}")

        logger.info(f"[DONE] Total images processed and saved: {len(processed_images)}")

    def stop(self):
        logger.info("Stopping Spark session...")
        self.spark.stop()
        logger.info("Spark session stopped.")

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    input_dir = r"C:\Users\user\Desktop\Keshav\Shillong_Fire_MAPPING"
    output_dir = r"C:\Users\user\Desktop\Keshav\Notes"

    processor = DistributedImageProcessor()
    try:
        processor.process_images(input_dir, output_dir)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        processor.stop()
