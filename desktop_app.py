import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import os
import threading
import io

# Import existing logic
from ml import model, gradcam

class EcoScanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EcoScan AI - Desktop")
        self.root.geometry("900x700")
        
        # Dark Theme Colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#4CAF50" # Green for healthy
        self.card_bg = "#2d2d2d"
        self.button_bg = "#3d3d3d"
        
        self.root.configure(bg=self.bg_color)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.current_image_bytes = None
        
        self._setup_ui()
        
        # Load model in background
        self.status_label.config(text=f"Loading model on {self.device}...", fg="yellow")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill="x", padx=20, pady=20)
        
        title = tk.Label(header_frame, text="EcoScan AI", font=("Segoe UI", 24, "bold"), bg=self.bg_color, fg=self.fg_color)
        title.pack(side="left")
        
        subtitle = tk.Label(header_frame, text="Coral Bleaching Detection", font=("Segoe UI", 14), bg=self.bg_color, fg="#aaaaaa")
        subtitle.pack(side="left", padx=10, pady=(10, 0))

        # Main Content Area
        main_content = tk.Frame(self.root, bg=self.bg_color)
        main_content.pack(expand=True, fill="both", padx=20, pady=10)
        
        # Image Display Area (Grid)
        main_content.columnconfigure(0, weight=1)
        main_content.columnconfigure(1, weight=1)
        
        # Left: Original Image
        self.left_frame = tk.Frame(main_content, bg=self.card_bg, padx=10, pady=10)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        tk.Label(self.left_frame, text="Original Image", bg=self.card_bg, fg=self.fg_color, font=("Segoe UI", 12)).pack()
        self.lbl_original = tk.Label(self.left_frame, bg=self.card_bg, text="No Image Selected", fg="#888888")
        self.lbl_original.pack(expand=True)
        
        # Right: GradCAM Analysis
        self.right_frame = tk.Frame(main_content, bg=self.card_bg, padx=10, pady=10)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        tk.Label(self.right_frame, text="AI Analysis (Grad-CAM)", bg=self.card_bg, fg=self.fg_color, font=("Segoe UI", 12)).pack()
        self.lbl_gradcam = tk.Label(self.right_frame, bg=self.card_bg, text="Waiting for prediction...", fg="#888888")
        self.lbl_gradcam.pack(expand=True)
        
        # Results Section
        self.result_frame = tk.Frame(self.root, bg=self.bg_color, pady=20)
        self.result_frame.pack(fill="x", padx=40)
        
        self.lbl_prediction = tk.Label(self.result_frame, text="", font=("Segoe UI", 18, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.lbl_prediction.pack()
        
        self.lbl_confidence = tk.Label(self.result_frame, text="", font=("Segoe UI", 12), bg=self.bg_color, fg="#bbbbbb")
        self.lbl_confidence.pack()

        # Bottom Controls
        control_frame = tk.Frame(self.root, bg=self.bg_color, pady=20)
        control_frame.pack(fill="x", side="bottom")
        
        self.btn_select = tk.Button(control_frame, text="Select Image", command=self.select_image, 
                                    font=("Segoe UI", 12), bg=self.button_bg, fg=self.fg_color, 
                                    relief="flat", padx=20, pady=10, state="disabled")
        self.btn_select.pack()
        
        self.status_label = tk.Label(self.root, text="Initializing...", bd=1, relief="sunken", anchor="w", bg="#333333", fg="white")
        self.status_label.pack(side="bottom", fill="x")

    def _load_model_thread(self):
        try:
            self.net = model.load_network(self.device)
            # Register hooks for GradCAM
            gradcam.register_hooks(self.net)
            
            # Update UI on main thread
            self.root.after(0, lambda: self.status_label.config(text=f"Model loaded ready on {self.device}", fg="#90ee90"))
            self.root.after(0, lambda: self.btn_select.config(state="normal", bg="#007acc"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model:\n{e}"))
            self.root.after(0, lambda: self.status_label.config(text="Model load failed", fg="red"))

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return
            
        try:
            with open(file_path, "rb") as f:
                self.current_image_bytes = f.read()
            
            # Display Original
            img = Image.open(io.BytesIO(self.current_image_bytes))
            img.thumbnail((400, 400)) # Resize for display
            self.photo_original = ImageTk.PhotoImage(img)
            self.lbl_original.config(image=self.photo_original, text="")
            
            # Reset Result
            self.lbl_gradcam.config(image="", text="Analyzing...")
            self.lbl_prediction.config(text="")
            self.lbl_confidence.config(text="")
            self.status_label.config(text="Processing image...", fg="yellow")
            
            # Run Prediction in Thread
            threading.Thread(target=self._process_image_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def _process_image_thread(self):
        if self.net is None:
            return

        try:
            # 1. Preprocess
            input_tensor = model.preprocess_image(self.current_image_bytes)
            
            # 2. Predict & Grad-CAM
            with torch.enable_grad():
                input_tensor = input_tensor.to(self.device).requires_grad_(True)
                
                # Forward
                outputs = self.net(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)
                
                label = model.CLASSES[predicted_idx.item()]
                conf_score = confidence.item()
                
                # Grad-CAM
                heatmap = gradcam.generate_gradcam(self.net, input_tensor, predicted_idx.item())
            
            # 3. Apply Heatmap
            heatmap_pil = gradcam.apply_heatmap(heatmap, self.current_image_bytes)
            
            # Update UI
            self.root.after(0, lambda: self._show_results(label, conf_score, heatmap_pil))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
            self.root.after(0, lambda: self.status_label.config(text="Prediction failed", fg="red"))

    def _show_results(self, label, confidence, heatmap_pil):
        # Display GradCAM
        heatmap_pil.thumbnail((400, 400))
        self.photo_gradcam = ImageTk.PhotoImage(heatmap_pil)
        self.lbl_gradcam.config(image=self.photo_gradcam, text="")
        
        # Display Text
        color = "#ff4444" if label.lower() == "bleached" else "#44ff44"
        self.lbl_prediction.config(text=f"Prediction: {label}", fg=color)
        self.lbl_confidence.config(text=f"Confidence: {confidence:.2%}")
        
        self.status_label.config(text="Analysis Complete", fg="white")

if __name__ == "__main__":
    root = tk.Tk()
    app = EcoScanApp(root)
    root.mainloop()
