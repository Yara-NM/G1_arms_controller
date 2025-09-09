import numpy as np
import cv2
import os
import glob
import json
import matplotlib.pyplot as plt
from datetime import datetime

class CameraIntrinsicCalibrator:
    def __init__(self, checkerboard_size=(7, 10), square_size=0.015, image_size=None):
        """
        Initialize the camera calibrator
        
        Args:
            checkerboard_size: (width, height) number of inner corners
            square_size: size of checkerboard squares in meters (0.04m = 40mm)
            image_size: (width, height) of images, will be detected automatically if None
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.image_size = image_size
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points from all images
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.images_used = []  # Track which images were successfully used
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
    def capture_images_from_camera(self, camera_id=4, num_images=20, save_dir="calibration_images"):
        """
        Capture calibration images from camera
        
        Args:
            camera_id: camera device ID
            num_images: number of images to capture
            save_dir: directory to save captured images
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera resolution if needed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        captured_count = 0
        print(f"Capturing {num_images} calibration images...")
        print("Press SPACE to capture image, 'q' to quit early")
        
        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Display frame
            display_frame = frame.copy()
            
            # Try to find checkerboard corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret_corners:
                # Draw corners on display frame
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners, ret_corners)
                cv2.putText(display_frame, "Checkerboard detected! Press SPACE to capture", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Move checkerboard into view", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"Captured: {captured_count}/{num_images}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Calibration Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret_corners:  # Space key and checkerboard detected
                filename = os.path.join(save_dir, f"calib_{captured_count:03d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Captured image {captured_count + 1}: {filename}")
                captured_count += 1
                
                # Brief pause to avoid multiple captures
                cv2.waitKey(500)
                
            elif key == ord('q'):  # Quit early
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Captured {captured_count} images in {save_dir}")
        return save_dir
    
    def load_images_from_directory(self, image_dir):
        """
        Load and process images from directory
        
        Args:
            image_dir: directory containing calibration images
        """
        # Get list of image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(image_files)} images")
        
        successful_detections = 0
        
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Failed to load image: {image_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Set image size from first successful image
            if self.image_size is None:
                self.image_size = gray.shape[::-1]  # (width, height)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                # Refine corner positions to subpixel accuracy
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                
                # Store object points and image points
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                self.images_used.append(image_path)
                successful_detections += 1
                print(f"  ✓ Checkerboard detected and corners refined")
                
                # Optional: Save image with detected corners
                # img_with_corners = img.copy()
                # cv2.drawChessboardCorners(img_with_corners, self.checkerboard_size, corners_refined, ret)
                # cv2.imwrite(f"detected_{i:03d}.jpg", img_with_corners)
                
            else:
                print(f"  ✗ Checkerboard not detected")
        
        print(f"\nSuccessfully detected checkerboard in {successful_detections}/{len(image_files)} images")
        
        if successful_detections < 10:
            print("Warning: Less than 10 successful detections. Consider capturing more images.")
        
        return successful_detections
    
    def calibrate_camera(self):
        """
        Perform camera calibration using collected corner points
        """
        if len(self.objpoints) < 10:
            raise ValueError("Need at least 10 successful corner detections for calibration")
        
        print(f"\nCalibrating camera with {len(self.objpoints)} images...")
        
        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            # Calculate reprojection error
            self.calibration_error = self.calculate_reprojection_error()
            
            print("✓ Camera calibration successful!")
            self.print_calibration_results()
            
            return True
        else:
            print("✗ Camera calibration failed!")
            return False
    
    def calculate_reprojection_error(self):
        """
        Calculate mean reprojection error
        """
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            # Project 3D points to image plane
            imgpoints_proj, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            
            # Calculate error for this image
            error = cv2.norm(self.imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            total_error += error * len(imgpoints_proj)
            total_points += len(imgpoints_proj)
        
        mean_error = total_error / total_points
        return mean_error
    
    def print_calibration_results(self):
        """
        Print calibration results
        """
        print("\n" + "="*50)
        print("CAMERA CALIBRATION RESULTS")
        print("="*50)
        
        print(f"Image size: {self.image_size[0]} x {self.image_size[1]}")
        print(f"Checkerboard: {self.checkerboard_size[0]} x {self.checkerboard_size[1]}")
        print(f"Square size: {self.square_size*1000:.1f} mm")
        print(f"Images used: {len(self.objpoints)}")
        print(f"Mean reprojection error: {self.calibration_error:.3f} pixels")
        
        print(f"\nCamera Matrix (K):")
        print(f"  fx = {self.camera_matrix[0,0]:.2f}")
        print(f"  fy = {self.camera_matrix[1,1]:.2f}")
        print(f"  cx = {self.camera_matrix[0,2]:.2f}")
        print(f"  cy = {self.camera_matrix[1,2]:.2f}")
        
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {self.dist_coeffs[0,0]:.6f}")
        print(f"  k2 = {self.dist_coeffs[0,1]:.6f}")
        print(f"  p1 = {self.dist_coeffs[0,2]:.6f}")
        print(f"  p2 = {self.dist_coeffs[0,3]:.6f}")
        print(f"  k3 = {self.dist_coeffs[0,4]:.6f}")
        
        # Field of view calculations
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        fov_x = 2 * np.arctan(self.image_size[0] / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(self.image_size[1] / (2 * fy)) * 180 / np.pi
        print(f"\nField of View:")
        print(f"  Horizontal: {fov_x:.1f}°")
        print(f"  Vertical: {fov_y:.1f}°")
        
        print("="*50)
    
    def save_calibration(self, filename="camera_calibration.json"):
        """
        Save calibration results to JSON file
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data to save. Run calibration first.")
        
        # Prepare calibration data
        calib_data = {
            'calibration_date': datetime.now().isoformat(),
            'image_size': {
                'width': int(self.image_size[0]),
                'height': int(self.image_size[1])
            },
            'checkerboard_size': {
                'width': self.checkerboard_size[0],
                'height': self.checkerboard_size[1]
            },
            'square_size_mm': float(self.square_size * 1000),
            'images_used': len(self.objpoints),
            'reprojection_error': float(self.calibration_error),
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'camera_parameters': {
                'fx': float(self.camera_matrix[0,0]),
                'fy': float(self.camera_matrix[1,1]),
                'cx': float(self.camera_matrix[0,2]),
                'cy': float(self.camera_matrix[1,2]),
                'k1': float(self.dist_coeffs[0,0]),
                'k2': float(self.dist_coeffs[0,1]),
                'p1': float(self.dist_coeffs[0,2]),
                'p2': float(self.dist_coeffs[0,3]),
                'k3': float(self.dist_coeffs[0,4])
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"\nCalibration saved to: {filename}")
        
        return filename
    
    def load_calibration(self, filename):
        """
        Load calibration from JSON file
        """
        with open(filename, 'r') as f:
            calib_data = json.load(f)
        
        self.camera_matrix = np.array(calib_data['camera_matrix'])
        self.dist_coeffs = np.array(calib_data['distortion_coefficients'])
        self.image_size = (calib_data['image_size']['width'], calib_data['image_size']['height'])
        self.calibration_error = calib_data['reprojection_error']
        
        print(f"Calibration loaded from: {filename}")
        self.print_calibration_results()
        
        return True
    
    def test_undistortion(self, test_image_path, save_result=True):
        """
        Test undistortion on a sample image
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data. Run calibration first.")
        
        img = cv2.imread(test_image_path)
        if img is None:
            raise ValueError(f"Could not load test image: {test_image_path}")
        
        # Undistort image
        undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs)
        
        if save_result:
            # Save comparison
            comparison = np.hstack([img, undistorted])
            cv2.imwrite('undistortion_comparison.jpg', comparison)
            print("Undistortion comparison saved to: undistortion_comparison.jpg")
        
        return undistorted

# Example usage and main function
def main():
    """
    Main calibration workflow
    """
    print("G1 Robot Head Camera Intrinsic Calibration")
    print("==========================================")
    
    # Initialize calibrator for 9x14 checkerboard with 40mm squares
    calibrator = CameraIntrinsicCalibrator(
        checkerboard_size=(7, 10),  # Inner corners
        square_size=0.015  # 40mm squares
    )
    
    # Option 1: Capture images from camera
    print("\nChoose calibration method:")
    print("1. Capture new images from camera")
    print("2. Use existing images from directory")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        # Capture images from camera
        print("\nStarting image capture...")
        print("Instructions:")
        print("- Hold checkerboard at different positions and orientations")
        print("- Ensure checkerboard fills different portions of the image")
        print("- Include various distances and angles")
        print("- Make sure checkerboard is flat and well-lit")
        
        image_dir = calibrator.capture_images_from_camera(
            camera_id=4,  # Change if using different camera
            num_images=25,  # Capture 25 images
            save_dir="calibration_640x480_images"
        )
        
    else:
        # Use existing images
        image_dir = input("Enter path to calibration images directory: ").strip()
        if not os.path.exists(image_dir):
            print(f"Directory not found: {image_dir}")
            return
    
    # Process images
    print(f"\nProcessing images from: {image_dir}")
    num_successful = calibrator.load_images_from_directory(image_dir)
    
    if num_successful < 10:
        print("Insufficient successful detections for calibration!")
        return
    
    # Perform calibration
    if calibrator.calibrate_camera():
        # Save results
        calib_file = calibrator.save_calibration("g1_camera_intrinsics.json")
        
        # Test undistortion if test image exists
        test_images = glob.glob(os.path.join(image_dir, "*"))
        if test_images:
            print("\nTesting undistortion...")
            try:
                calibrator.test_undistortion(test_images[0])
            except Exception as e:
                print(f"Could not test undistortion: {e}")
        
        print(f"\n✓ Calibration complete! Results saved to: {calib_file}")
        print("You can now use these intrinsics for extrinsic calibration.")
        
    else:
        print("\n✗ Calibration failed!")

if __name__ == "__main__":
    main()
