import cv2
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Optional
import torch
from ultralytics import YOLO
import os
import sys


class AutoYOLODetector:
    def __init__(self):
        # path
        self.model_path = "model.pt"

        self.source_path = "input/"

        self.output_dir = "detect_results/"

        # params
        self.confidence_threshold = 0.25  # conf
        self.iou_threshold = 0.40  # IoU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # classes
        self.target_classes = None
        self.save_json = True

        self.image_output_dir = Path(self.output_dir) / "images"
        self.label_output_dir = Path(self.output_dir) / "labels"
        self.video_output_dir = Path(self.output_dir) / "videos"
        self.json_output_dir = Path(self.output_dir) / "json_results"

        for directory in [self.image_output_dir, self.label_output_dir,
                          self.video_output_dir, self.json_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        self.colors = self._generate_colors(len(self.model.names))

        self.total_images = 0
        self.total_videos = 0
        self.total_detections = 0
        self.start_time = time.time()

        print(f"load complete！device: {self.device}")
        print(f"classes number: {len(self.model.names)}")
        print(f"output path: {self.output_dir}")
        print("=" * 50)

    def _generate_colors(self, n: int) -> List[tuple]:

        colors = []
        np.random.seed(52)
        for _ in range(n):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors

    def _draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:

        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_id = det['class_id']
            cls_name = det['class_name']

            color = self.colors[cls_id % len(self.colors)]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(annotated,
                          (x1, y1 - text_height - baseline - 10),
                          (x1 + text_width, y1),
                          color, -1)

            cv2.putText(annotated, label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated

    def _save_yolo_format(self, detections: List[dict], label_path: Path,
                          image_width: int, image_height: int):

        with open(label_path, 'w', encoding='utf-8') as f:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cls_id = det['class_id']
                conf = det['confidence']


                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height


                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _save_json_results(self, results: dict, json_path: Path):

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def detect_image(self, image_path: Path) -> dict:

        print(f"image detecting: {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  error {image_path}")
            return None

        height, width = image.shape[:2]

        results = self.model(image, conf=self.confidence_threshold,
                             iou=self.iou_threshold, classes=self.target_classes)

        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                detections.append(detection)
                self.total_detections += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = image_path.stem

        if detections:
            annotated_image = self._draw_detections(image, detections)
            output_image_path = self.image_output_dir / f"{base_name}_detected.jpg"
            cv2.imwrite(str(output_image_path), annotated_image)
        else:
            output_image_path = self.image_output_dir / f"{base_name}_no_detections.jpg"
            cv2.imwrite(str(output_image_path), image)

        label_path = self.label_output_dir / f"{base_name}.txt"
        self._save_yolo_format(detections, label_path, width, height)

        result = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'output_image': str(output_image_path),
            'label_file': str(label_path),
            'detections_count': len(detections),
            'detections': detections,
            'image_size': {'width': width, 'height': height},
            'detection_time': datetime.now().isoformat()
        }

        if self.save_json:
            json_path = self.json_output_dir / f"{base_name}.json"
            self._save_json_results(result, json_path)

        print(f"  complete:  {len(detections)} targets")
        return result

    def detect_video(self, video_path: Path):

        print(f"video detecting: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  error {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  information: {width}x{height}, FPS: {fps:.2f}, total frames: {total_frames}")

        base_name = video_path.stem
        output_video_path = self.video_output_dir / f"{base_name}_detected.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        video_label_dir = self.label_output_dir / base_name
        video_label_dir.mkdir(exist_ok=True)

        all_detections = []
        frame_count = 0
        frame_detections = []

        print("  starting video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.confidence_threshold,
                                 iou=self.iou_threshold, classes=self.target_classes)

            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': cls_name
                    }
                    detections.append(detection)
                    self.total_detections += 1

            if detections:
                annotated_frame = self._draw_detections(frame, detections)
                out.write(annotated_frame)
            else:
                out.write(frame)

            frame_label_path = video_label_dir / f"frame_{frame_count:06d}.txt"
            self._save_yolo_format(detections, frame_label_path, width, height)

            frame_result = {
                'frame_number': frame_count,
                'detections_count': len(detections),
                'detections': detections
            }
            all_detections.append(frame_result)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f" {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

        result = {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'output_video': str(output_video_path),
            'label_directory': str(video_label_dir),
            'total_frames': frame_count,
            'total_detections': sum([len(f['detections']) for f in all_detections]),
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height
            },
            'detection_time': datetime.now().isoformat()
        }

        if self.save_json:
            json_path = self.json_output_dir / f"{base_name}.json"
            self._save_json_results(result, json_path)

        print(f" {frame_count} frames, {result['total_detections']} target")
        return result

    def process_directory(self):

        source_path = Path(self.source_path)

        if not source_path.exists():
            print(f"error  {self.source_path}")
            return

        print(f"start: {self.source_path}")

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']

        image_files = []
        video_files = []

        if source_path.is_file():

            suffix = source_path.suffix.lower()
            if suffix in image_extensions:
                image_files.append(source_path)
            elif suffix in video_extensions:
                video_files.append(source_path)
            else:
                print(f"unsupported: {suffix}")
        else:

            for ext in image_extensions:
                image_files.extend(source_path.rglob(f"*{ext}"))
                image_files.extend(source_path.rglob(f"*{ext.upper()}"))

            for ext in video_extensions:
                video_files.extend(source_path.rglob(f"*{ext}"))
                video_files.extend(source_path.rglob(f"*{ext.upper()}"))

        print(f" {len(image_files)} images , {len(video_files)} video")
        print("-" * 50)

        image_results = []
        if image_files:
            print("processing images...")
            for i, img_file in enumerate(image_files, 1):
                print(f"[{i}/{len(image_files)}] ", end="")
                result = self.detect_image(img_file)
                if result:
                    image_results.append(result)
                self.total_images += 1
            print("done！")

        video_results = []
        if video_files:
            print("\nprocessing videos...")
            for i, video_file in enumerate(video_files, 1):
                print(f"[{i}/{len(video_files)}] ", end="")
                result = self.detect_video(video_file)
                if result:
                    video_results.append(result)
                self.total_videos += 1
            print("done！")

        self._generate_summary_report(image_results, video_results)

    def _generate_summary_report(self, image_results, video_results):

        elapsed_time = time.time() - self.start_time
        class_counts = {}
        all_results = image_results + video_results

        for result in all_results:
            if 'detections' in result:
                for det in result['detections']:
                    cls_name = det['class_name']
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            elif 'frames' in result:
                for frame in result['frames']:
                    for det in frame['detections']:
                        cls_name = det['class_name']
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        report = f"""
{'=' * 60}
YOLO report
{'=' * 60}

time: {elapsed_time:.1f} s
number: {self.total_images} images, {self.total_videos} videos
total: {self.total_detections} target

structure list:
  {self.output_dir}/
    ├── images/      
    ├── labels/      
    ├── videos/      
    └── json_results/

result:
"""

        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

        for cls_name, count in sorted_classes:
            report += f"  {cls_name}: {count} 个\n"

        if not class_counts:
            report += "  no target\n"

        report += f"""
param settings:
  model: {self.model_path}
  conf: {self.confidence_threshold}
  IoU: {self.iou_threshold}
  device: {self.device}
  classes: {self.target_classes if self.target_classes else 'all'}

{'=' * 60}
results saved in: {os.path.abspath(self.output_dir)}
{'=' * 60}
"""

        print(report)

        report_path = Path(self.output_dir) / "detection_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"report saved in: {report_path}")


def main():

    print("YOLO detecting...")
    print("loading...")

    try:
        detector = AutoYOLODetector()

        detector.process_directory()

        print("\ndone！press any key to continue...")
        if sys.platform == "win32":
            os.system("pause")

    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        print("\npress any key to continue...")
        if sys.platform == "win32":
            os.system("pause")


if __name__ == "__main__":
    main()