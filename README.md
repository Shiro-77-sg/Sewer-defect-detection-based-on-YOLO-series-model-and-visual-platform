🚀 Sewer Pipeline Defect Detection System(Demo)

![Screenshot_2](https://img.shields.io/badge/YOLO-v8-00FFFF?style=flat-square)
![Screenshot_2](https://img.shields.io/badge/Python-3.8%252B-blue?style=flat-square)
![Screenshot_2](https://img.shields.io/badge/PyTorch-2.1%252B-red?style=flat-square)
![Screenshot_2](https://img.shields.io/badge/OpenCV-4.9%252B-green?style=flat-square)

📖 项目简介 / Project Introduction
中文 | English

这是一个基于YOLOv8的智能化下水管道缺陷检测系统。系统能够自动检测并分类下水管道中的6种常见缺陷类型（变形、障碍物、破裂、断开、错位、沉积），提供完整的训练、推理和可视化平台解决方案。

English      

This is an intelligent sewer pipeline defect detection system based on YOLOv8. The system can automatically detect and classify 6 common types of defects in sewer pipelines (Deformation, Obstacle, Rupture, Disconnect, Misalignment, Deposition), providing a complete solution for training, inference, and visualization platform.

✨ 主要特性 / Key Features

🎯 检测能力 / Detection Capabilities
6种缺陷类型检测 / 6 Defect Types Detection:

变形 (Deformation)
障碍物 (Obstacle)
破裂 (Rupture)
断开 (Disconnect)
错位 (Misalignment)
沉积 (Deposition)

🔧 技术特性 / Technical Features
先进网络架构: 基于MCFN-YOLO的改进模型

多模态输入: 支持图像、视频和实时摄像头检测

完整流程: 从数据准备、模型训练到推理部署的全流程支持

可视化界面: 提供用户友好的GUI操作平台

缺陷评估: 基于面积的严重程度评估和解决方案建议

📁 项目结构 / Project Structure
``````
project/
├── config.yaml              # 数据集配置文件
├── model.yaml              # 模型架构文件
├── train.py                # 模型训练脚本
├── Detect.py               # 批量推理脚本
├── detect_box_evaluate.py  # 缺陷等级评估脚本
├── input/                 # 检测输入
    ├── images/             # 检测图像
    ├── videos/               # 检测视频
├── detect_results/        # 检测结果
    ├── images/             # 输出图像
    ├── videos/               # 输出视频
    ├── json_results/          # json结果
    ├── labels/                   #txt结果
    ├── detection_report.txt        #检测报告
├── platform_demo.py        # 可视化演示平台
├── logs/                 # 平台结果日志
    ├── images/             # 输出图像
    ├── json/                   # json结果
├── requirements_demo.txt   # 环境依赖文件
└── data/                   # 数据集目录
    ├── train/             # 训练集
    ├── val/               # 验证集
    └── test/              # 测试集
``````

⚙️ 环境安装
安装依赖
``````
# 安装基本依赖
pip install -r requirements_demo.txt
``````
🚀 快速开始
1. 数据准备
确保数据集按以下结构组织：
``````
# config.yaml配置示例
train: D:/.../data/train
val: D:/.../data/val
test: D:/.../data/test
nc: 6
names: 
  0: Deformation
  1: Obstacle
  2: Rupture
  3: Disconnect
  4: Misalignment
  5: Deposition
``````
2. 模型训练
``````
# train.py 训练配置
from ultralytics import YOLO
# 从零开始训练
model = YOLO("model.yaml")  
# 使用预训练权重
# model = YOLO("yolov8s.pt")
# 开始训练
model.train(data="config.yaml", epochs=300, batch=16, lr0=0.01, optimizer='SGD')
``````
运行训练：
``````
python train.py
``````
3. 批量推理
``````
# 将待检测文件放入input/目录
# 运行检测脚本
python Detect.py

# 结果将保存在detect_results/目录
``````

4. 缺陷评估(Demo)
``````
# detect_box_evaluate.py 使用示例
python detect_box_evaluate.py

# 或指定视频文件
# 修改代码中的视频路径
``````
🧪 缺陷评估

基于缺陷面积自动评估严重程度

严重程度分级：低风险、中风险、高风险

5. 可视化平台(Demo)
``````
# 启动GUI演示平台
python platform_demo.py
``````
🖥️ 可视化平台Demo版(Demo版仅做实时性演示，检测数据效果不代表最终版)

核心功能(Demo版本只支持.pt格式模型，后续更新.onnx与.trt格式的支持)

📁 模型加载: 支持.pt格式YOLO模型

🖼️ 图像检测: 单张/批量图像处理

🎥 视频检测: 视频文件分析和实时处理

📷 实时摄像头: 支持摄像头实时检测

📊 数据统计: 实时缺陷统计和可视化(缺陷统计仅做演示，统计效果不代表最终版)

📝 日志记录: 完整检测日志保存(日志报告仅做演示，不代表最终效果)

⚠️ 缺陷评估: 严重程度分级和解决方案建议(Demo版内不包含)

操作流程
1. 加载模型 → 2. 选择数据源 → 3. 开始检测 → 4. 查看结果 → 5. 保存结果

🧪 模型架构

MCFN-YOLO

多尺度特征融合网络 (Multi-Scale Context Fusion Network)

全连接路径聚合网络 (Fully Connected Path Aggregation Network)

高效多尺度注意力机制 (Efficient Multi-scale Attention)

自适应空间感知检测头 ( Adaptive Scale-Aware Detection Head)

技术优势: 🎯 更高的检测精度; 🔄 更好的多尺度适应性

📊 参数说明

训练参数
``````
epochs: 300
batch_size: 16
learning_rate: 0.01
optimizer: SGD
device: CUDA
``````
推理参数
``````
confidence_threshold: 0.25
iou_threshold: 0.40
target_classes: None (所有类别)
save_format: JSON/TXT/Images
``````
📁 文件说明

核心文件
``````
文件	        说明
config.yaml	数据集路径和类别配置
model.yaml	MCFN-YOLO网络架构定义
train.py	模型训练脚本
Detect.py	自动化批量检测脚本
platform_demo.py	可视化演示平台
detect_box_evaluate.py	缺陷评估脚本
``````
输出目录
``````
detect_results/
├── images/          # 标注后的图像
├── labels/          # YOLO格式标签
├── videos/          # 处理后的视频
├── json_results/    # JSON格式结果
└── detection_report.txt  # 检测报告

评估输出/
├── video_output.avi # 标注评估后的视频
└── 评估日志.txt      # 缺陷评估日志
``````
🔧 自定义配置

1.修改检测阈值
``````
# 在Detect.py中修改
self.confidence_threshold = 0.25  # 置信度阈值
self.iou_threshold = 0.45         # IoU阈值
``````
2.调整评估阈值
``````
# 在detect_box_evaluate.py中修改
# 修改面积阈值
if area > 100:  # 可调整的面积阈值
    # 处理逻辑

# 修改严重程度阈值
def get_severity_and_solution(defect_type, area):
    if defect_type == "Deformation":
        if area < 150:  # 调整阈值
            return "Low", "Deformation: Routine inspection"
        # ...
``````

3.选择特定类别
``````
# 只检测特定缺陷类型
self.target_classes = [0, 2, 5]  # 变形、破裂、沉积
``````

4.调整保存选项
``````
self.save_json = True      # 保存JSON结果
self.save_labels = True    # 保存标签文件
self.save_images = True    # 保存标注图像
``````
🐛 故障排除

常见问题

1.模型加载失败
``````
检查模型文件路径

确认PyTorch版本兼容性
``````
2.CUDA内存不足
``````
# 减小batch_size
model.train(batch=8)
``````
3.检测结果不准确
``````
调整置信度阈值

检查训练数据质量
``````
4.缺陷评估不准确
``````
调整面积阈值
``````
5.GUI平台卡顿
``````
# 降低检测频率
self.detect_interval = 2  # 每2帧检测一次
``````
🤝 贡献感谢

🙏 致谢

感谢Ultralytics团队提供的YOLOv8框架

感谢所有开源贡献者的支持

特别感谢数据标注团队的辛勤工作

📄 许可证

本项目采用MIT许可证。 

- 查看 LICENSE 文件了解详情。
