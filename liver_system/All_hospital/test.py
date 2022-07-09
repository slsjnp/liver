import PIL.Image as Image
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms

import cv2
import time
from All_hospital.munet import MUnet
from All_hospital.Data_preprocessing import read_data

def model_test(raw_path, model_path, save_path, reconstructed_image_path, flag=1):
    # flag = 1 is liver and liver-tumor, flag = 2 is spleen

    raw, flinames = read_data(raw_path)

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    y_transforms = transforms.ToTensor()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MUnet(1, 1)
    # model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    for i in range(raw.shape[0]):
        img_x = raw[i]

        save_img = img_x.copy()
        save_img = Image.fromarray(save_img)
        save_img = save_img.convert('RGB')
        save_img = np.asarray(save_img)

        img_x = Image.fromarray(np.int16(np.array(img_x)))
        h, w = img_x.size
        img_x = img_x.resize((256, 256))
        img_x = img_x.convert('L')

        # spleen
        if (flag == 2):
            img_x = np.asarray(img_x)
            img_x = np.rot90(img_x, k=-1)
            img_x = Image.fromarray(img_x)

        img_x = x_transforms(img_x)
        img_x = img_x.to(device)
        img_x = torch.unsqueeze(img_x, 0)

        img = model(img_x)

        trann = transforms.ToPILImage()

        img = torch.squeeze(img)
        img = img.detach().cpu().numpy()

        img[img >= 0.5] = 255
        img[img < 0.5] = 0
        img = img.astype(np.uint8)

        # spleen
        if (flag == 2):
            img = np.rot90(img)

        img = trann(img)
        img = img.convert('L')
        img = img.resize((h, w))

        # name = flinames[i][:-4] + ".bmp"
        name = "{}".format(i) + ".bmp"
        rip = os.path.join(reconstructed_image_path, name)
        img.save(rip)

        img = np.asarray(img)
        ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(save_img, contours, -1, (255, 0, 0), 1)

        save_img = Image.fromarray(save_img)
        save_img = save_img.convert('RGB')
        # name = flinames[i][:-4] + ".bmp"
        name = "{}".format(i) + ".bmp"
        sp = os.path.join(save_path, name)
        save_img.save(sp)


#    torch.cuda.empty_cache()
# if __name__ == '__main__':
#     path = "/home/sj/workspace/my_git/liver/liver_system/domo/app/static/f58b800cbec311eba8fc85e136c9fc6b/dcm/"
#     save = "/home/sj/workspace/my_git/liver/liver_system/domo/app/static/f58b800cbec311eba8fc85e136c9fc6b/liver/bmp/"
#     model_path = "/home/sj/workspace/my_git/liver/liver_system/domo/app/static/ckp/liver.pth"
#     rip_path = "/home/sj/workspace/my_git/liver/liver_system/domo/app/static/f58b800cbec311eba8fc85e136c9fc6b/liver/cut_bmp/"
#     model_test(path, model_path, save, rip_path, 1)
