"""
Workshop #7: Face Detection using Haar Cascade Algorithm
Learning Outcomes:
- Understand object detection, face detection, and Haar Cascade algorithm
- Implement face detection program with Haar Cascade
- Create graphical interface for user interaction
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading

class HaarCascadeFaceDetector:
    """
    Implementation of Haar Cascade algorithm for face detection
    """
    
    def __init__(self):
        # Load pre-trained Haar cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check if cascade loaded successfully
        if self.face_cascade.empty():
            raise Exception("Could not load Haar cascade classifier")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image using Haar Cascade algorithm
        
        Args:
            image: Input image (BGR format)
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size, smaller objects are ignored
            
        Returns:
            List of face rectangles [(x, y, w, h), ...]
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def calculate_haar_features(self, image, x, y, w, h):
        """
        Calculate basic Haar-like features for a face region
        This demonstrates the concept behind Haar Cascade
        """
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return None
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic Haar-like features
        features = {}
        
        # Feature 1: Eye region vs cheek region (horizontal)
        height, width = gray_face.shape
        upper_half = gray_face[:height//2, :]
        lower_half = gray_face[height//2:, :]
        
        features['eye_cheek_diff'] = np.mean(upper_half) - np.mean(lower_half)
        
        # Feature 2: Center vs sides (vertical)
        left_third = gray_face[:, :width//3]
        center_third = gray_face[:, width//3:2*width//3]
        right_third = gray_face[:, 2*width//3:]
        
        features['center_side_diff'] = np.mean(center_third) - (np.mean(left_third) + np.mean(right_third))/2
        
        # Feature 3: Mouth region detection
        mouth_region = gray_face[2*height//3:, width//4:3*width//4]
        features['mouth_intensity'] = np.mean(mouth_region)
        
        return features

class FaceDetectionApp:
    """
    Main application class with GUI for face detection
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Workshop #7: Face Detection using Haar Cascade")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize face detector
        try:
            self.face_detector = HaarCascadeFaceDetector()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize face detector: {str(e)}")
            return
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.detected_faces = []
        self.processing = False
        
        # Setup GUI
        self.setup_gui()
        
        # Status
        self.update_status("Ready. Please select an image to detect faces.")
    
    def setup_gui(self):
        """Setup the graphical user interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Detection using Haar Cascade Algorithm", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Buttons
        self.load_button = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        
        self.detect_button = ttk.Button(control_frame, text="Detect Faces", command=self.detect_faces)
        self.detect_button.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        self.detect_button.config(state='disabled')
        
        self.clear_button = ttk.Button(control_frame, text="Clear Results", command=self.clear_results)
        self.clear_button.grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(control_frame, text="Detection Parameters", padding="5")
        params_frame.grid(row=3, column=0, pady=10, sticky=tk.W+tk.E)
        
        # Scale factor
        ttk.Label(params_frame, text="Scale Factor:").grid(row=0, column=0, sticky=tk.W)
        self.scale_var = tk.DoubleVar(value=1.1)
        scale_scale = ttk.Scale(params_frame, from_=1.05, to=1.3, variable=self.scale_var, 
                               orient=tk.HORIZONTAL, length=150)
        scale_scale.grid(row=0, column=1, pady=2)
        self.scale_label = ttk.Label(params_frame, text="1.1")
        self.scale_label.grid(row=0, column=2, sticky=tk.W)
        scale_scale.configure(command=lambda v: self.scale_label.config(text=f"{float(v):.2f}"))
        
        # Min neighbors
        ttk.Label(params_frame, text="Min Neighbors:").grid(row=1, column=0, sticky=tk.W)
        self.neighbors_var = tk.IntVar(value=5)
        neighbors_scale = ttk.Scale(params_frame, from_=3, to=8, variable=self.neighbors_var, 
                                   orient=tk.HORIZONTAL, length=150)
        neighbors_scale.grid(row=1, column=1, pady=2)
        self.neighbors_label = ttk.Label(params_frame, text="5")
        self.neighbors_label.grid(row=1, column=2, sticky=tk.W)
        neighbors_scale.configure(command=lambda v: self.neighbors_label.config(text=str(int(float(v)))))
        
        # Min size
        ttk.Label(params_frame, text="Min Size:").grid(row=2, column=0, sticky=tk.W)
        self.min_size_var = tk.IntVar(value=30)
        min_size_scale = ttk.Scale(params_frame, from_=20, to=100, variable=self.min_size_var, 
                                  orient=tk.HORIZONTAL, length=150)
        min_size_scale.grid(row=2, column=1, pady=2)
        self.min_size_label = ttk.Label(params_frame, text="30")
        self.min_size_label.grid(row=2, column=2, sticky=tk.W)
        min_size_scale.configure(command=lambda v: self.min_size_label.config(text=str(int(float(v)))))
        
        # Results frame
        results_frame = ttk.LabelFrame(control_frame, text="Detection Results", padding="5")
        results_frame.grid(row=4, column=0, pady=10, sticky=tk.W+tk.E)
        
        self.results_text = tk.Text(results_frame, height=8, width=30, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Image display area
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="10")
        self.image_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg='white', width=600, height=400)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        
        # Configure canvas grid
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure control frame grid
        control_frame.columnconfigure(0, weight=1)
        params_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image using OpenCV
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return
                
                self.current_image = self.original_image.copy()
                self.detected_faces = []
                
                # Display image
                self.display_image()
                
                # Enable detect button
                self.detect_button.config(state='normal')
                
                # Update status
                height, width = self.original_image.shape[:2]
                self.update_status(f"Image loaded: {os.path.basename(file_path)} ({width}x{height})")
                
                # Clear results
                self.clear_results_text()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def detect_faces(self):
        """Detect faces in the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.processing:
            return
        
        # Start detection in a separate thread to prevent GUI freezing
        self.processing = True
        self.detect_button.config(state='disabled', text="Detecting...")
        self.update_status("Processing... Please wait.")
        
        thread = threading.Thread(target=self._detect_faces_thread)
        thread.daemon = True
        thread.start()
    
    def _detect_faces_thread(self):
        """Face detection thread"""
        try:
            # Get parameters
            scale_factor = self.scale_var.get()
            min_neighbors = self.neighbors_var.get()
            min_size = (self.min_size_var.get(), self.min_size_var.get())
            
            # Detect faces
            faces = self.face_detector.detect_faces(
                self.original_image,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size
            )
            
            # Store results
            self.detected_faces = faces
            
            # Update GUI in main thread
            self.root.after(0, self._update_detection_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
            self.root.after(0, self._reset_detection_button)
    
    def _update_detection_results(self):
        """Update GUI with detection results"""
        try:
            # Create image with detected faces
            result_image = self.original_image.copy()
            
            # Clear results text
            self.clear_results_text()
            
            if len(self.detected_faces) == 0:
                self.results_text.insert(tk.END, "No faces detected.\n\n")
                self.results_text.insert(tk.END, "Try adjusting the parameters:\n")
                self.results_text.insert(tk.END, "- Lower scale factor for more detection\n")
                self.results_text.insert(tk.END, "- Lower min neighbors for more detection\n")
                self.results_text.insert(tk.END, "- Smaller min size for detecting smaller faces\n")
            else:
                self.results_text.insert(tk.END, f"Detected {len(self.detected_faces)} face(s):\n\n")
                
                # Draw rectangles and analyze each face
                for i, (x, y, w, h) in enumerate(self.detected_faces):
                    # Draw rectangle
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add face number
                    cv2.putText(result_image, f"Face {i+1}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Calculate Haar features
                    features = self.face_detector.calculate_haar_features(result_image, x, y, w, h)
                    
                    # Add to results
                    self.results_text.insert(tk.END, f"Face {i+1}:\n")
                    self.results_text.insert(tk.END, f"  Position: ({x}, {y})\n")
                    self.results_text.insert(tk.END, f"  Size: {w}x{h}\n")
                    
                    if features:
                        self.results_text.insert(tk.END, f"  Haar Features:\n")
                        self.results_text.insert(tk.END, f"    Eye-Cheek Diff: {features['eye_cheek_diff']:.2f}\n")
                        self.results_text.insert(tk.END, f"    Center-Side Diff: {features['center_side_diff']:.2f}\n")
                        self.results_text.insert(tk.END, f"    Mouth Intensity: {features['mouth_intensity']:.2f}\n")
                    
                    self.results_text.insert(tk.END, "\n")
            
            # Update current image
            self.current_image = result_image
            
            # Display updated image
            self.display_image()
            
            # Update status
            if len(self.detected_faces) > 0:
                self.update_status(f"Detection complete: {len(self.detected_faces)} face(s) found")
            else:
                self.update_status("Detection complete: No faces found")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update results: {str(e)}")
        
        finally:
            self._reset_detection_button()
    
    def _reset_detection_button(self):
        """Reset detect button state"""
        self.processing = False
        self.detect_button.config(state='normal', text="Detect Faces")
    
    def clear_results(self):
        """Clear detection results"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.detected_faces = []
            self.display_image()
            self.clear_results_text()
            self.update_status("Results cleared")
    
    def clear_results_text(self):
        """Clear results text area"""
        self.results_text.delete(1.0, tk.END)
    
    def display_image(self):
        """Display current image on canvas"""
        if self.current_image is None:
            return
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not initialized yet, try again later
            self.root.after(100, self.display_image)
            return
        
        # Calculate scaling to fit image in canvas
        img_height, img_width = self.current_image.shape[:2]
        
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_image = cv2.resize(self.current_image, (new_width, new_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Center image on canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        # Store image position and scale for mouse events
        self.image_x = x
        self.image_y = y
        self.image_scale = scale
        self.displayed_width = new_width
        self.displayed_height = new_height
    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if self.current_image is None:
            return
        
        # Convert canvas coordinates to image coordinates
        image_x = int((event.x - self.image_x) / self.image_scale)
        image_y = int((event.y - self.image_y) / self.image_scale)
        
        # Check if click is within image bounds
        img_height, img_width = self.current_image.shape[:2]
        if 0 <= image_x < img_width and 0 <= image_y < img_height:
            # Check if click is on a detected face
            for i, (x, y, w, h) in enumerate(self.detected_faces):
                if x <= image_x <= x+w and y <= image_y <= y+h:
                    messagebox.showinfo("Face Info", 
                                      f"Clicked on Face {i+1}\n"
                                      f"Position: ({x}, {y})\n"
                                      f"Size: {w}x{h}\n"
                                      f"Click coordinates: ({image_x}, {image_y})")
                    return
            
            # If not on a face, show pixel information
            pixel_value = self.current_image[image_y, image_x]
            messagebox.showinfo("Pixel Info", 
                              f"Coordinates: ({image_x}, {image_y})\n"
                              f"BGR Value: {pixel_value}")
    
    def on_canvas_motion(self, event):
        """Handle canvas mouse motion"""
        if self.current_image is None:
            return
        
        # Convert canvas coordinates to image coordinates
        image_x = int((event.x - self.image_x) / self.image_scale)
        image_y = int((event.y - self.image_y) / self.image_scale)
        
        # Check if within image bounds
        img_height, img_width = self.current_image.shape[:2]
        if 0 <= image_x < img_width and 0 <= image_y < img_height:
            # Check if hovering over a detected face
            for i, (x, y, w, h) in enumerate(self.detected_faces):
                if x <= image_x <= x+w and y <= image_y <= y+h:
                    self.canvas.configure(cursor="hand2")
                    return
            
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(cursor="arrow")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = FaceDetectionApp(root)
    
    # Handle window closing
    def on_closing():
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()