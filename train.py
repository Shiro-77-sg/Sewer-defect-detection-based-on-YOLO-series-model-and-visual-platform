from ultralytics import YOLO
if __name__ == '__main__':
    # load model
    model = YOLO("model.yaml")  # new start
    #model.load('last.pt')
    #model = YOLO("yolov8s.pt")  # pretrained model

    # Use the model
    results = model.train(data="config.yaml", epochs=300, batch=16, lr0=0.01, amp=False, optimizer='SGD')  # train