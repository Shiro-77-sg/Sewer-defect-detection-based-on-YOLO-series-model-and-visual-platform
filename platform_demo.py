import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import os
from datetime import datetime
import time
import queue
import json
import numpy as np


class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sewer Pipeline Defect Detection Platform Demo")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # color
        self.bg_color = "#2c3e50"
        self.fg_color = "#ecf0f1"
        self.accent_color = "#3498db"
        self.warning_color = "#e74c3c"
        self.success_color = "#2ecc71"

        # fontsize
        self.title_font = ("Helvetica", 16, "bold")
        self.text_font = ("Helvetica", 12)
        self.small_font = ("Helvetica", 10)

        # windows
        self.root.configure(bg=self.bg_color)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=self.text_font, padding=6)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TFrame', background=self.bg_color)
        style.configure('TNotebook', background=self.bg_color)
        style.configure('TNotebook.Tab', background=self.bg_color, foreground=self.fg_color)

        # load
        self.model = None
        self.model_type = None
        self.model_name = "No Model Loaded"
        self.defect_counts = {}
        self.box_details = []
        self.current_image = None
        self.current_images = []
        self.video_capture = None
        self.is_video_playing = False
        self.video_thread = None
        self.is_camera = False

        # classes
        self.class_names = []

        # param
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.detect_interval = 1
        self.frame_counter = 0
        self.last_detected_frame = None
        self.last_detected_result = None

        # fps
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.processing_fps = 0
        self.display_fps = 0

        # control
        self.processing_thread = None
        self.display_thread = None
        self.stop_threads = False
        self.image_detection_thread = None

        # log
        self.log_file = None
        self.log_start_time = 0
        self.last_log_time = 0
        self.log_interval = 10  # (s)
        self.detection_session_id = None
        self.log_counter = 0
        self.total_frames_processed = 0
        self.total_defects_detected = 0

        # 5s save
        self.last_save_time = 0
        self.save_interval = 5

        # detect para
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        self.create_main_layout()

        self.create_status_bar()

    def create_main_layout(self):

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # left
        control_panel = ttk.Frame(main_container, width=250, style='TFrame')
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        # right
        display_panel = ttk.Frame(main_container, style='TFrame')
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_control_panel(control_panel)
        self.create_display_panel(display_panel)

    def create_control_panel(self, parent):

        # title
        title_label = ttk.Label(parent, text="Detection Platform", font=self.title_font)
        title_label.pack(pady=(0, 20))

        # model
        model_frame = ttk.LabelFrame(parent, text="Load Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)

        self.model_status = ttk.Label(model_frame, text="No Model Selected", foreground="#e74c3c")
        self.model_status.pack(pady=5)


        select_model_btn = ttk.Button(model_frame, text="Load YOLO Model", command=self.select_model)
        select_model_btn.pack(fill=tk.X, pady=5)


        file_frame = ttk.LabelFrame(parent, text="Select", padding=10)
        file_frame.pack(fill=tk.X, pady=5)


        select_image_btn = ttk.Button(file_frame, text="Image", command=self.select_image)
        select_image_btn.pack(fill=tk.X, pady=2)

        select_video_btn = ttk.Button(file_frame, text="Video", command=self.select_video_file)
        select_video_btn.pack(fill=tk.X, pady=2)

        select_camera_btn = ttk.Button(file_frame, text="Camera", command=self.select_camera)
        select_camera_btn.pack(fill=tk.X, pady=2)


        control_frame = ttk.LabelFrame(parent, text="Detection Control", padding=10)
        control_frame.pack(fill=tk.X, pady=5)

        self.detect_btn = ttk.Button(control_frame, text="Start Detecting", state=tk.DISABLED,
                                     command=self.start_detection)
        self.detect_btn.pack(fill=tk.X, pady=2)

        self.video_control_btn = ttk.Button(control_frame, text="Continue/Pause", state=tk.DISABLED,
                                            command=self.toggle_video_playback)
        self.video_control_btn.pack(fill=tk.X, pady=2)


        stats_frame = ttk.LabelFrame(parent, text="Detection Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.stats_text = tk.Text(stats_frame, height=10, wrap=tk.WORD, bg="#34495e", fg=self.fg_color,
                                  font=self.small_font, padx=5, pady=5)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert(tk.END, "Waiting Results...")
        self.stats_text.config(state=tk.DISABLED)

        save_btn = ttk.Button(stats_frame, text="Save Results", command=self.save_results)
        save_btn.pack(fill=tk.X, pady=(10, 0))

    def create_display_panel(self, parent):

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.original_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.original_tab, text="Original Scene")
        self.original_canvas = tk.Canvas(self.original_tab, bg="#34495e")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="Detection Results")
        self.result_canvas = tk.Canvas(self.result_tab, bg="#34495e")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        self.details_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.details_tab, text="More Information")
        self.details_text = tk.Text(self.details_tab, wrap=tk.WORD, bg="#34495e", fg=self.fg_color,
                                    font=self.text_font, padx=10, pady=10)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.insert(tk.END, "Detection Information in Here...")
        self.details_text.config(state=tk.DISABLED)


        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Log Records")
        self.log_text = tk.Text(self.log_tab, wrap=tk.WORD, bg="#34495e", fg=self.fg_color,
                                font=self.small_font, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(self.log_tab, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.insert(tk.END, "Log records will appear here...\n")
        self.log_text.config(state=tk.DISABLED)

    def create_status_bar(self):

        status_bar = ttk.Frame(self.root, height=25)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(status_bar, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        version_label = ttk.Label(status_bar, text="Sewer Pipeline Defect Detection Platform Demo",
                                  relief=tk.SUNKEN, anchor=tk.E)
        version_label.pack(fill=tk.X, side=tk.RIGHT)

    def select_model(self):

        filetypes = [
            ("PyTorch Model", "*.pt"),
            ("All files", "*.*")
        ]

        model_path = filedialog.askopenfilename(filetypes=filetypes)
        if model_path:
            try:

                file_ext = os.path.splitext(model_path)[1].lower()

                if file_ext == '.pt':
                    self.model = YOLO(model_path)

                    test_img = Image.new('RGB', (640, 640))
                    self.model.predict(source=test_img, device='0', half=True, verbose=False)
                    self.model_type = 'pt'

                    if hasattr(self.model, 'names') and self.model.names:
                        self.class_names = list(self.model.names.values())
                    else:
                        self.class_names = [f"class_{i}" for i in range(80)]
                    self.log_message(f"PyTorch Model Load Successfully: {model_path}")

                else:
                    raise ValueError(f"Unsupported model format: {file_ext}")

                self.model_name = os.path.basename(model_path)
                self.model_status.config(text=f"Loaded: {self.model_name} ({self.model_type.upper()})",
                                         foreground=self.success_color)
                self.detect_btn.config(state=tk.NORMAL)
                self.update_status(f"{self.model_type.upper()} Model Load Successfully")

            except Exception as e:
                self.update_status(f"Model Load Failed: {str(e)}", error=True)
                self.log_message(f"Model Load Failed: {str(e)}", error=True)

    def predict_with_model(self, image):

        if self.model_type == 'pt':

            results = self.model.predict(image, device='0', half=True, verbose=False)
            return results
        else:
            raise ValueError("No model loaded")

    def start_detection(self):

        if self.current_images:

            threading.Thread(target=self.detect_images, daemon=True).start()
        elif self.video_capture:
            if not self.is_video_playing:
                self.is_video_playing = True
                self.frame_count = 0
                self.start_time = time.time()
                self.frame_counter = 0
                self.stop_threads = False


                self.last_save_time = time.time()


                self.init_logging()


                while not self.frame_queue.empty(): self.frame_queue.get_nowait()
                while not self.result_queue.empty(): self.result_queue.get_nowait()

                self.processing_thread = threading.Thread(target=self.process_video_frames, daemon=True)
                self.display_thread = threading.Thread(target=self.display_video_frames, daemon=True)
                self.processing_thread.start()
                self.display_thread.start()

    def detect_images(self):

        total_images = len(self.current_images)

        if total_images == 0:
            messagebox.showwarning("No Images", "Please select images first.")
            return

        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first.")
            return


        images_log_file = None
        try:

            if not os.path.exists('logs'):
                os.makedirs('logs')

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            images_log_filename = f"logs/images_detection_{timestamp}.log"
            images_log_file = open(images_log_filename, 'a', encoding='utf-8')


            start_msg = f"\n{'=' * 60}\n"
            start_msg += f"Image Detection Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            start_msg += f"Total Images: {total_images}\n"
            start_msg += f"Model: {self.model_name} ({self.model_type.upper()})\n"
            start_msg += f"Classes: {len(self.class_names)}\n"
            start_msg += f"{'=' * 60}\n"

            images_log_file.write(start_msg)
            images_log_file.flush()
            self.append_to_log_viewer(start_msg)

            processed_count = 0

            for i, image_path in enumerate(self.current_images):
                try:

                    progress = (i + 1) / total_images * 100
                    self.root.after(0, self.update_status,
                                    f"Processing image {i + 1}/{total_images} ({progress:.1f}%)")


                    img = cv2.imread(image_path)
                    if img is None:
                        error_msg = f"Failed to read image: {image_path}\n"
                        images_log_file.write(error_msg)
                        self.append_to_log_viewer(error_msg)
                        continue


                    self.root.after(0, self.display_frame, img.copy(), True)


                    results = self.predict_with_model(img)

                    if results is None:
                        error_msg = f"Prediction failed for: {image_path}\n"
                        images_log_file.write(error_msg)
                        self.append_to_log_viewer(error_msg)
                        continue

                    self.process_results(results)

                    result_img = results[0].plot()
                    self.root.after(0, self.display_frame, result_img, False)

                    self.root.after(0, self.update_stats)
                    self.root.after(0, self.update_details)

                    image_name = os.path.basename(image_path)
                    log_entry = self.create_image_log_entry(image_name, i + 1)

                    images_log_file.write(log_entry)
                    images_log_file.flush()

                    self.append_to_log_viewer(log_entry)

                    self.auto_save_image_results(image_path, img, result_img, results)

                    processed_count += 1

                    time.sleep(0.1)

                except Exception as e:
                    error_msg = f"Error processing {image_path}: {str(e)}\n"
                    images_log_file.write(error_msg)
                    self.append_to_log_viewer(error_msg)

            end_msg = f"\n{'=' * 60}\n"
            end_msg += f"Image Detection Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            end_msg += f"Processed: {processed_count}/{total_images} images\n"
            end_msg += f"Results saved to: {images_log_filename}\n"
            end_msg += f"{'=' * 60}\n"

            images_log_file.write(end_msg)
            images_log_file.flush()
            self.append_to_log_viewer(end_msg)

            self.root.after(0, self.update_status,
                            f"Image detection completed: {processed_count}/{total_images} images")
            messagebox.showinfo("Image Detection Complete",
                                f"Processed {processed_count}/{total_images} images.\nResults saved to: {images_log_filename}")

        except Exception as e:
            self.log_message(f"Image detection error: {str(e)}", error=True)
        finally:
            if images_log_file:
                images_log_file.close()

    def create_image_log_entry(self, image_name, image_num):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_defects = sum(stats["count"] for stats in self.defect_counts.values())

        log_entry = f"\n[Image #{image_num}] {timestamp}\n"
        log_entry += f"Image: {image_name}\n"
        log_entry += f"Total Defects: {total_defects}\n"

        for defect_name, stats in self.defect_counts.items():
            log_entry += f"  {defect_name}: {stats['count']} (avg conf: {stats['avg_confidence']:.3f})\n"

        log_entry += "-" * 40 + "\n"

        if self.box_details:
            log_entry += f"Detection Boxes ({len(self.box_details)}):\n"
            for i, box in enumerate(self.box_details, 1):
                log_entry += f"  Box {i}: {box['class_name']} (conf: {box['confidence']:.4f})\n"

        return log_entry

    def select_image(self):

        image_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if image_paths:
            self.reset_video()
            self.current_image = None
            self.current_images = list(image_paths)

            if self.current_images:
                self.display_image(self.current_images[0])
                self.update_status(f"Selected {len(self.current_images)} image(s)")

                self.append_to_log_viewer(f"\nSelected {len(self.current_images)} image(s):\n")
                for i, path in enumerate(self.current_images, 1):
                    self.append_to_log_viewer(f"  {i}. {os.path.basename(path)}\n")

            self.detect_btn.config(state=tk.NORMAL)

            self.result_canvas.delete("all")

    def auto_save_image_results(self, image_path, original_img, result_img, results):
        try:

            if not os.path.exists('logs/images'):
                os.makedirs('logs/images', exist_ok=True)
            if not os.path.exists('logs/json'):
                os.makedirs('logs/json', exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.basename(image_path).split('.')[0]

            result_img_path = f"logs/images/{image_name}_result_{timestamp}.jpg"
            cv2.imwrite(result_img_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            json_path = f"logs/json/image_detection_{image_name}_{timestamp}.json"

            results_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "result_image_path": result_img_path,
                "defect_counts": self.convert_to_serializable(self.defect_counts),
                "box_details": self.convert_to_serializable(self.box_details),
                "model_type": self.model_type,
                "model_name": self.model_name,
                "classes": self.class_names
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            self.log_message(f"Image results saved: {json_path} and {result_img_path}")

        except Exception as e:
            self.log_message(f"Failed to auto-save image results: {str(e)}", error=True)

    def process_video_frames(self):

        while self.is_video_playing and not self.stop_threads:
            try:
                ret, frame = self.video_capture.read()
                if not ret:

                    self.log_periodic_stats(final=True)
                    break

                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0: self.fps = self.frame_count / elapsed_time


                if self.frame_queue.qsize() < 25:
                    self.frame_queue.put(frame)
                time.sleep(0.005)
            except:
                break

    def display_video_frames(self):

        current_time = time.time()
        self.last_log_time = current_time
        self.last_save_time = current_time

        while self.is_video_playing and not self.stop_threads:
            try:

                frame = None
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()

                if frame is not None:
                    self.total_frames_processed += 1

                    self.root.after(0, self.display_frame, frame.copy(), True)

                    results = self.predict_with_model(frame)

                    if results is not None:

                        result_img = results[0].plot()

                        self.process_results(results)
                        self.root.after(0, self.update_stats)
                        self.root.after(0, self.update_details)

                        self.last_detected_result = result_img
                        self.root.after(0, self.display_frame, result_img, False)

                        current_time = time.time()
                        if current_time - self.last_save_time >= self.save_interval:
                            self.auto_save_video_frame(frame, result_img, results)
                            self.last_save_time = current_time

                    self.root.after(0, self.update_status,
                                    f"Video Analyzing...   FPS: {self.fps:.1f} frames per second")

                    current_time = time.time()
                    if current_time - self.last_log_time >= self.log_interval:
                        self.log_periodic_stats()
                        self.last_log_time = current_time

                time.sleep(0.001)
            except queue.Empty:
                continue
            except:
                break

    def auto_save_video_frame(self, original_frame, result_frame, results):
        try:

            if not os.path.exists('logs/images'):
                os.makedirs('logs/images', exist_ok=True)
            if not os.path.exists('logs/json'):
                os.makedirs('logs/json', exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            source_type = "camera" if self.is_camera else "video"

            original_path = f"logs/images/{source_type}_original_{timestamp}.jpg"
            cv2.imwrite(original_path, original_frame)

            result_path = f"logs/images/{source_type}_result_{timestamp}.jpg"
            cv2.imwrite(result_path, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

            json_path = f"logs/json/{source_type}_detection_{timestamp}.json"

            results_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "source_type": source_type,
                "frame_number": self.total_frames_processed,
                "original_image_path": original_path,
                "result_image_path": result_path,
                "defect_counts": self.convert_to_serializable(self.defect_counts),
                "box_details": self.convert_to_serializable(self.box_details),
                "model_type": self.model_type,
                "model_name": self.model_name,
                "classes": self.class_names,
                "fps": self.fps,
                "save_interval": self.save_interval
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            self.log_message(f"{source_type.capitalize()} frame saved: {json_path} and {result_path}")

        except Exception as e:
            self.log_message(f"Failed to auto-save video frame: {str(e)}", error=True)

    def log_periodic_stats(self, final=False):

        if not self.log_file:
            return

        self.log_counter += 1
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = time.time() - self.log_start_time

        total_defects = sum(stats["count"] for stats in self.defect_counts.values())
        self.total_defects_detected += total_defects

        if final:
            log_msg = f"\n{'=' * 60}\n"
            log_msg += f"Detection Session Ended: {current_time}\n"
            log_msg += f"Session ID: {self.detection_session_id}\n"
        else:
            log_msg = f"\n{'=' * 60}\n"
            log_msg += f"Periodic Log #{self.log_counter}: {current_time}\n"

        log_msg += f"Elapsed Time: {elapsed_time:.1f} seconds\n"
        log_msg += f"Total Frames Processed: {self.total_frames_processed}\n"
        log_msg += f"Average FPS: {self.fps:.2f}\n"
        log_msg += f"Total Defects Detected: {total_defects}\n"
        log_msg += f"Cumulative Defects: {self.total_defects_detected}\n"
        log_msg += f"Model Type: {self.model_type.upper()}\n"
        log_msg += f"Save Interval: {self.save_interval} seconds\n"

        if self.defect_counts:
            log_msg += f"\nDefect Statistics:\n"
            for defect_name, stats in self.defect_counts.items():
                log_msg += f"  - {defect_name}:\n"
                log_msg += f"    Count: {stats['count']}\n"
                log_msg += f"    Avg Confidence: {stats['avg_confidence']:.3f}\n"
        else:
            log_msg += f"\nNo defects detected in this period.\n"

        if final:
            log_msg += f"\nSession Summary:\n"
            log_msg += f"  Total Duration: {elapsed_time:.1f} seconds\n"
            log_msg += f"  Total Frames: {self.total_frames_processed}\n"
            log_msg += f"  Overall Average FPS: {self.total_frames_processed / elapsed_time:.2f}\n"
            log_msg += f"  Total Defects: {self.total_defects_detected}\n"
            log_msg += f"{'=' * 60}\n"
        else:
            log_msg += f"{'=' * 60}\n"

        self.log_file.write(log_msg)
        self.log_file.flush()

        self.append_to_log_viewer(log_msg)

        if final and self.log_file:
            self.log_file.close()
            self.log_file = None

    def append_to_log_viewer(self, message):

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def process_results(self, results):

        self.defect_counts.clear()
        self.box_details.clear()

        for result in results:
            for box in result.boxes:
                name = result.names[int(box.cls)]
                conf = float(box.conf)

                xyxy = box.xyxy[0].cpu().numpy() if box.xyxy.is_cuda else box.xyxy[0].numpy()
                self.process_single_box(name, conf, xyxy)

        for defect_name in self.defect_counts:
            if self.defect_counts[defect_name]["count"] > 0:
                self.defect_counts[defect_name]["avg_confidence"] = float(
                    self.defect_counts[defect_name]["total_confidence"] /
                    self.defect_counts[defect_name]["count"]
                )
                self.defect_counts[defect_name]["avg_area"] = float(
                    self.defect_counts[defect_name]["total_area"] /
                    self.defect_counts[defect_name]["count"]
                )

                self.defect_counts[defect_name]["total_confidence"] = float(
                    self.defect_counts[defect_name]["total_confidence"])
                self.defect_counts[defect_name]["total_area"] = float(self.defect_counts[defect_name]["total_area"])

    def process_single_box(self, name, conf, xyxy):

        x1, y1, x2, y2 = xyxy

        width = x2 - x1
        height = y2 - y1
        area = width * height

        if name not in self.defect_counts:
            self.defect_counts[name] = {
                "count": 0,
                "total_confidence": 0,
                "avg_confidence": 0,
                "avg_area": 0,
                "total_area": 0
            }

        self.defect_counts[name]["count"] += 1
        self.defect_counts[name]["total_confidence"] += conf
        self.defect_counts[name]["total_area"] += area

        self.box_details.append({
            "class_name": name,
            "confidence": float(conf),
            "width": float(width),
            "height": float(height),
            "area": float(area),
            "position": (float(x1), float(y1), float(x2), float(y2))
        })

    def convert_to_serializable(self, obj):

        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def update_stats(self):

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        if not self.defect_counts:
            self.stats_text.insert(tk.END, "No defects detected\n")
        else:
            self.stats_text.insert(tk.END, f"Model: {self.model_name} ({self.model_type.upper()})\n")
            self.stats_text.insert(tk.END, f"Classes: {len(self.class_names)}\n\n")
            for defect_name, stats in self.defect_counts.items():
                self.stats_text.insert(tk.END,
                                       f"• {defect_name}:\n"
                                       f"  Count: {stats['count']}\n"
                                       f"  Avg Confidence: {stats['avg_confidence']:.3f}\n"
                                       f"  Avg Area: {stats['avg_area']:.1f} px²\n"
                                       f"  Total Area: {stats['total_area']:.1f} px²\n\n"
                                       )

        self.stats_text.config(state=tk.DISABLED)

    def update_details(self):

        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)

        if not self.box_details:
            self.details_text.insert(tk.END, "No detection boxes found\n")
        else:
            self.details_text.insert(tk.END, f"Detection Box Details ({len(self.box_details)} boxes):\n")
            self.details_text.insert(tk.END, "=" * 50 + "\n")

            for i, box in enumerate(self.box_details, 1):
                self.details_text.insert(tk.END,
                                         f"Box #{i}:\n"
                                         f"  Class: {box['class_name']}\n"
                                         f"  Confidence: {box['confidence']:.4f}\n"
                                         f"  Position: ({box['position'][0]:.1f}, {box['position'][1]:.1f}) - "
                                         f"({box['position'][2]:.1f}, {box['position'][3]:.1f})\n"
                                         f"  Size: {box['width']:.1f} × {box['height']:.1f}\n"
                                         f"  Area: {box['area']:.1f} px²\n"
                                         f"-" * 30 + "\n"
                                         )

        self.details_text.config(state=tk.DISABLED)

    def select_video_file(self):

        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if video_path:
            self.reset_video()
            self.clear_image_state()
            self.is_camera = False
            self.video_capture = cv2.VideoCapture(video_path)
            self.detect_btn.config(state=tk.NORMAL)
            self.video_control_btn.config(state=tk.NORMAL)

            self.original_canvas.delete("all")
            self.result_canvas.delete("all")

            # update
            self.update_status(f"Video selected: {os.path.basename(video_path)}")
            self.log_message(f"Video selected: {video_path}")

    def select_camera(self):

        self.reset_video()
        self.clear_image_state()
        self.is_camera = True
        self.video_capture = cv2.VideoCapture(0)
        if self.video_capture.isOpened():
            self.detect_btn.config(state=tk.NORMAL)
            self.video_control_btn.config(state=tk.NORMAL)

            self.original_canvas.delete("all")
            self.result_canvas.delete("all")

            self.update_status("Camera selected")
            self.log_message("Camera selected")
        else:
            messagebox.showerror("Error", "Cannot open camera!")
            self.log_message("Failed to open camera", error=True)

    def clear_image_state(self):

        self.current_image = None
        self.current_images = []
        self.defect_counts.clear()
        self.box_details.clear()

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Waiting Results...")
        self.stats_text.config(state=tk.DISABLED)

        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "Detection Information in Here...")
        self.details_text.config(state=tk.DISABLED)

    def toggle_video_playback(self):

        self.is_video_playing = not self.is_video_playing
        if self.is_video_playing:
            self.update_status("Video playing...")
        else:
            self.update_status("Video paused")

    def display_frame(self, frame, original=True):

        if frame is None:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = self.original_canvas if original else self.result_canvas
        img_pil = Image.fromarray(frame)

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400

        img_pil.thumbnail((canvas_width, canvas_height))
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=img_tk)
        canvas.image = img_tk

    def display_image(self, path):

        img = cv2.imread(path)
        if img is not None:
            self.display_frame(img, original=True)

    def save_results(self):

        if not self.defect_counts and not self.box_details:
            messagebox.showwarning("No Data", "No detection results to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:

                results_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "defect_counts": self.defect_counts,
                    "box_details": self.box_details,
                    "total_frames_processed": self.total_frames_processed,
                    "total_defects_detected": self.total_defects_detected,
                    "session_id": self.detection_session_id,
                    "model_type": self.model_type,
                    "model_name": self.model_name,
                    "classes": self.class_names
                }

                results_data = self.convert_to_serializable(results_data)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)

                self.log_message(f"Results saved to: {file_path}")
                messagebox.showinfo("Success", f"Results saved successfully to:\n{file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                self.log_message(f"Failed to save results: {str(e)}", error=True)

    def update_status(self, message, error=False):

        self.status_label.config(text=message, foreground=self.warning_color if error else self.fg_color)

    def log_message(self, message, error=False):

        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "[ERROR]" if error else "[INFO]"
        log_msg = f"{timestamp} {prefix} {message}\n"

        if self.log_file:
            self.log_file.write(log_msg)
            self.log_file.flush()

        self.append_to_log_viewer(log_msg)

    def init_logging(self):

        if not os.path.exists('logs'):
            os.makedirs('logs')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_type = "camera" if self.is_camera else "video"
        self.detection_session_id = f"{source_type}_{timestamp}"

        log_filename = f"logs/detection_{self.detection_session_id}.log"
        self.log_file = open(log_filename, 'a', encoding='utf-8')

        start_msg = f"\n{'=' * 60}\n"
        start_msg += f"Detection Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        start_msg += f"Session ID: {self.detection_session_id}\n"
        start_msg += f"Source: {'Camera' if self.is_camera else 'Video File'}\n"
        start_msg += f"Model: {self.model_name} ({self.model_type.upper()})\n"
        start_msg += f"Classes: {len(self.class_names)}\n"
        start_msg += f"Save Interval: {self.save_interval} seconds\n"
        start_msg += f"{'=' * 60}\n"

        self.log_file.write(start_msg)
        self.log_file.flush()

        self.append_to_log_viewer(start_msg)

        self.log_start_time = time.time()
        self.last_log_time = self.log_start_time
        self.last_save_time = self.log_start_time
        self.log_counter = 0
        self.total_frames_processed = 0
        self.total_defects_detected = 0

    def reset_video(self):

        self.is_video_playing = False
        self.stop_threads = True

        if self.log_file and self.detection_session_id:
            self.log_periodic_stats(final=True)

        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def on_closing(self):

        self.reset_video()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()