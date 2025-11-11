# Distributed Parallel Processing of Drone-Captured Images for Orthomosaic Generation using Apache Spark


## 🌟 Overview

This project presents a scalable and robust architectural solution for generating **high-resolution orthomosaics** (panoramas) from large aerial image datasets captured by drone platforms. It leverages the **Apache Spark** framework to distribute computationally intensive computer vision tasks across a cluster, effectively overcoming the memory and processing bottlenecks of traditional single-machine stitching software.

The system is implemented in Python using **PySpark** and the **OpenCV** library.


<!-- First Row -->
<p align="center">
    <img src="Output/3.jpg" alt="Screenshot 1" width="200" />
    <img src="Output/6 (2).jpg" />
</p>

## 🚀 Key Features

*   **Horizontal Scalability:** Distributes the most demanding stages—Feature Extraction and Feature Matching—across a Spark cluster, enabling efficient handling of datasets too large for a single workstation.
    
*   **Robust Feature Detection:** Utilizes a multi-detector strategy, combining **SIFT**, **ORB**, and **AKAZE** algorithms to ensure robust feature correspondence across diverse image conditions.
    
*   **Pipeline Resilience:** Features intelligent fallback mechanisms, including **Skip-Ahead Logic** to bypass failed matches and **Affine Transformation Fallback** for geometrically unstable image pairs, ensuring chain continuity.
    
*   **High-Fidelity Output:** Employs RANSAC for robust homography estimation, followed by warping, feather-blending, and automatic cropping to produce seamless, coherent panoramas.
    
*   **Optional Post-Processing:** Integrates optional enhancement filters (e.g., CLAHE contrast enhancement, sharpening) to improve the visual quality of the final output.
    

## 🛠️ System Architecture & Workflow

The pipeline is a hybrid design, strategically balancing distributed and local computation:

| Stage | Execution Location | Purpose |
| --- | --- | --- |
| 1. Image Loading & Prep | Distributed (Executors) | Loads images, resizes them (mitigates OOM errors), and encodes them into Base64 strings for efficient Spark transmission. |
| 2. Feature Extraction | Distributed (Executors) | Massively parallel task to extract SIFT, ORB, and AKAZE keypoints and descriptors from image partitions. |
| 3. Feature Matching | Distributed (Executors) | Parallel matching of adjacent image pairs, applying Lowe's Ratio Test and cross-checking for robustness. |
| 4. Homography Estimation | Centralized (Driver) | Calculates robust transformation matrices (Homographies or Affine fallbacks) for global image alignment using RANSAC. |
| 5. Warping & Blending | Centralized (Driver) | Creates the final canvas, warps images, and applies feather blending to create a seamless panorama. |
| 6. Post-Processing | Centralized (Driver) | Optional contrast (CLAHE) and detail (sharpening) enhancements. |

## ⚙️ Setup and Configuration Guide

### Prerequisites

1.  **Java:** Java 11 or a compatible version (Spark runs on the JVM).
    
2.  **Python:** Python 3.10 or later.
    
3.  **Apache Spark:** Download a pre-built version of Apache Spark.
    

### Installation Steps

1.  **Set Environment Variables:**
    
    *   Set `SPARK_HOME` to your Spark installation directory (e.g., `/opt/spark` or `C:\Spark`).
        
    *   Add `$SPARK_HOME/bin` to your system `PATH`.
        
    *   **(Windows Only):** Install `winutils.exe` and set `HADOOP_HOME` to ensure proper filesystem interaction.
        
    *   Set `PYSPARK_PYTHON` to your Python interpreter path.
        
2.  **Install Python Dependencies:** The core application requires PySpark, OpenCV, and NumPy.
    
        pip install pyspark numpy opencv-python
        
        
        
        
    
3.  **Local Spark Cluster Setup (Development/Testing)**
    
    **Start the Master Node:**
    
        # Linux/macOS
        $SPARK_HOME/sbin/start-master.sh
        
        # Windows (Command Prompt)
        C:\Spark\bin\spark-class org.apache.spark.deploy.master.Master
        
        
        
        
    
    _Note the Master URL (e.g., `spark://<your-hostname>:7077`). You can monitor the UI at `http://localhost:8080`._
    
    **Start a Worker Node (Connect to the Master URL):**
    
        # Linux/macOS
        $SPARK_HOME/sbin/start-worker.sh spark://<your-hostname>:7077
        
        # Windows (Command Prompt)
        C:\Spark\bin\spark-class org.apache.spark.deploy.worker.Worker spark://<your-hostname>:7077
        
        
        
        
    

### Running the Application

Once the Spark cluster is running (even in local mode), the application can be submitted:

    # Example command:
    spark-submit --master spark://<your-hostname>:7077 \
                 --executor-cores 12 \
                 --executor-memory 12G \
                 path/to/your_stitching_script.py \
                 --input_dir /path/to/drone/images \
                 --output_file final_orthomosaic.png
    
    
    
    

## 🔮 Future Work and Enhancements

1.  **Integrate GPS Metadata:** Enhance the loader to extract EXIF GPS data from drone images for rough initial placement and reducing unnecessary matching operations.
    
2.  **Migrate to Spark DataFrames:** Refactor the RDD-based implementation to use the Spark DataFrame API to leverage the Catalyst optimizer and Tungsten execution engine for performance gains.
    
3.  **Implement Global Optimization:** Integrate a **Bundle Adjustment** stage (e.g., using `scipy.optimize`) after initial homography calculation to minimize cumulative error (drift) and achieve globally consistent stitching results.
    
4.  **Develop a User Interface (GUI):** Create a web-based dashboard (using frameworks like Streamlit or Flask) to improve usability, allowing for easier parameter tuning, image upload, and result viewing.
