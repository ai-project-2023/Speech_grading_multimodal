import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evaluation using a model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--video', dest='video_path', help='Path of the video file to use.',
        default='./datasets/file.mp4', type=str)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video_path = args.video_path  # 영상 파일 경로
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened correctly
    if not cap.isOpened():
        print("Cannot open the video file.")
        exit()

    output_dir = "./output_frames"  # 이미지 저장 디렉토리
    frame_count = 0
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break

            start_fps = time.time()

            faces = detector(frame)
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).cuda(gpu)
                    img = img.unsqueeze(0)

                    # Gaze prediction
                    gaze_pitch, gaze_yaw = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

                    print("Pitch:", pitch_predicted)
                    print("Yaw:", yaw_predicted)

                    # Draw gaze direction on the frame
                    # frame = draw_gaze(frame, landmarks, pitch_predicted, yaw_predicted)

            # myFPS = 1.0 / (time.time() - start_fps)
            # cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #             (0, 255, 0), 1, cv2.LINE_AA)

            # # Save the frame as an image
            # output_path = f"{output_dir}/frame_{frame_count:04d}.jpg"
            # cv2.imwrite(output_path, frame)
            # frame_count += 1

            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    cap.release()
    cv2.destroyAllWindows()
