# yolov8
from ultralytics import YOLO

if __name__ ==  '__main__':
    # # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    #
    # # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format

    # # Load a model
    # model = YOLO("yolov8x.yaml")  # build a new model from scratch
    # model = model.to('cuda')
    # # Use the model
    # results = model.train(data="config.yaml", epochs=300)  # train the model

    # #validation
    # model = YOLO("runs/detect/train7/weights/best.pt")
    # results = model.val()


    # #make prediction
    # model = YOLO("runs/detect/train8/weights/best.pt")
    # results = model.predict(source="./kirin-pale-ale169.jpeg", imgsz=640, conf=0.5, save=False)
    # print('hdgfdgdf')
    # ##results[0].boxes
    # ##results[0].boxes.cls
    # ##results[0].boxes.conf
    # # results = model.predict(source="./test_predict/images", imgsz=640, conf=0.5, save=True)

    # model = YOLO("runs/detect/train8/weights/best.pt")
    model = YOLO("./best.pt")
    import os
    import shutil
    path_img = "./prediction"
    dir_list = os.walk(path_img)
    for dirpath, dirname_list, filename_list in dir_list:
        # create bottle, tin-can, unknown folders
        if 'bottle' not in dirpath and 'tin-can' not in dirpath and 'unknown' not in dirpath:
            if len(filename_list) > 0:
                bottle_folder = os.path.join(dirpath, 'bottle')  # 0
                tin_can_folder = os.path.join(dirpath, 'tin-can')  # 1
                unknown_folder = os.path.join(dirpath, 'unknown')  # unknown class
                if not os.path.exists(bottle_folder):
                    os.makedirs(bottle_folder)
                if not os.path.exists(tin_can_folder):
                    os.makedirs(tin_can_folder)
                if not os.path.exists(unknown_folder):
                    os.makedirs(unknown_folder)

            for file_name in filename_list:
                file_name_path = os.path.join(dirpath, file_name)
                if file_name.endswith('.svg'):
                    os.remove(file_name_path)
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    results = model.predict(source=file_name_path, conf=0.5, save=False)

                    # no detection
                    if len(results[0].boxes.cls) == 0:
                        if not file_name.startswith('unknown_'):
                            shutil.move(file_name_path, os.path.join(unknown_folder, 'unknown_' + file_name))
                        else:
                            shutil.move(file_name_path, os.path.join(unknown_folder, file_name))
                        #print('moved file to unknown folder')
                    else:
                        # bottle class with highest probability
                        if results[0].boxes.cls[0] == 0:
                            if not file_name.startswith('bottle_'):
                                shutil.move(file_name_path, os.path.join(bottle_folder, 'bottle_' + file_name))
                            else:
                                shutil.move(file_name_path, os.path.join(bottle_folder, file_name))
                            #print('moved file to bottle folder')
                        # tin-can class with highest probability
                        elif results[0].boxes.cls[0] == 1:
                            if not file_name.startswith('tin-can_'):
                                shutil.move(file_name_path, os.path.join(tin_can_folder, 'tin-can_' + file_name))
                            else:
                                shutil.move(file_name_path, os.path.join(tin_can_folder, file_name))
                            #print('moved file to tin-can folder')



