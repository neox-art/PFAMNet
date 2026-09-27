import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse
from losses import *
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import measure
from scipy.spatial import distance


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")
from skimage.measure import label, regionprops

def calculate_accuracy(output, mask):
    """
    计算像素级指标 (OA, F1, IoU) 和对象级指标 (GOC, GUC, GTC)
    output, mask: numpy array, 二值图 0/1
    """
    # ---------------- Pixel-level ----------------
    if len(mask.shape) != 1:
        mask_flat = mask.reshape(-1)
    else:
        mask_flat = mask
    if len(output.shape) != 1:
        output_flat = output.reshape(-1)
    else:
        output_flat = output

    mask_flat = mask_flat.astype(np.uint8)
    output_flat = output_flat.astype(np.uint8)

    # 混淆矩阵
    matrix = confusion_matrix(mask_flat, output_flat, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    OA = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    P = tp / (tp + fp) if (tp + fp) != 0 else 0
    R = tp / (tp + fn) if (tp + fn) != 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
    IOU = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    # ---------------- Object-level ----------------
    def label_objects(mask_inner):
        return label(mask_inner, connectivity=1)

    labeled_pred = label_objects(output)
    labeled_gt = label_objects(mask)

    props_pred = regionprops(labeled_pred)
    props_gt = regionprops(labeled_gt)

    OC, UC, TC, area_list = [], [], [], []

    for Si in props_pred:
        area_Si = Si.area
        max_overlap = 0
        best_Oi = None
        for Oi in props_gt:
            overlap = np.logical_and(labeled_pred == Si.label, labeled_gt == Oi.label).sum()
            if overlap > max_overlap:
                max_overlap = overlap
                best_Oi = Oi
        if best_Oi is None:
            oc, uc, tc = 1.0, 1.0, 1.0
        else:
            area_Oi = best_Oi.area
            inter = max_overlap
            oc = 1 - inter / (area_Oi + 1e-12)
            uc = 1 - inter / (area_Si + 1e-12)
            tc = np.sqrt((oc ** 2 + uc ** 2) / 2)
        OC.append(oc)
        UC.append(uc)
        TC.append(tc)
        area_list.append(area_Si)

    area_sum = np.sum(area_list) + 1e-10
    if area_sum == 0:
        GOC = 1.0
        GUC = 1.0
        GTC = 1.0
    else:
        GOC = np.sum(np.array(OC) * np.array(area_list)) / area_sum
        GUC = np.sum(np.array(UC) * np.array(area_list)) / area_sum
        GTC = np.sum(np.array(TC) * np.array(area_list)) / area_sum

    return OA, F1, IOU, GOC, GUC, GTC
def evaluate(device, epoch, model, data_loader, writer,model_type):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        if model_type in ["field","bfinet","reaunet"]:
            criterion = BCEDiceLoss()
        if model_type in ["bsinet"]:
            criterion =LossMulti(num_classes=2)
        for iter, data in enumerate(tqdm(data_loader)):
            inputs = data[1].to(device)
            targets = [x.to(device) for x in data[2:]]
            outputs = model(inputs)
            loss  = criterion(outputs[0],targets[0])
            # loss = F.nll_loss(outputs[0], targets.squeeze(1))
            losses.append(loss.item())

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start

def get_pred_mask(output):
    """
    output: [B, C, H, W]
    C=2 -> argmax
    C=1 -> sigmoid + threshold
    return: [B, H, W]
    """
    if output.shape[1] == 2:
        return torch.argmax(output, dim=1)
    elif output.shape[1] == 1:
        return (torch.sigmoid(output) > 0.5).long().squeeze(1)
    else:
        raise ValueError(f"Unsupported output shape: {output.shape}")
def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    valid_OA, valid_F1, valid_IOU = [], [], []
    valid_GOC, valid_GUC, valid_GTC = [], [], []

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            

            inputs = data[1].to(device)

            targets = [x.to(device) for x in data[2:]]
            outputs = model(inputs)

            # 二值化
            pred_bin = get_pred_mask(outputs[0])             # [B,H,W]
            label_bin = (targets[0] > 0).long().squeeze(1)       # [B,H,W]

            pred_np = pred_bin.cpu().numpy().astype(np.uint8)
            label_np = label_bin.cpu().numpy().astype(np.uint8)

            # 每张图单独算
            for b in range(pred_np.shape[0]):
                OA, F1, IOU, GOC, GUC, GTC = calculate_accuracy(pred_np[b], label_np[b])
                valid_OA.append(OA)
                valid_F1.append(F1)
                valid_IOU.append(IOU)
                valid_GOC.append(GOC)
                valid_GUC.append(GUC)
                valid_GTC.append(GTC)

           
            print(
                f"OA={np.mean(valid_OA):.4f}, "
                f"F1={np.mean(valid_F1):.4f}, "
                f"IoU={np.mean(valid_IOU):.4f}, "
                f"GOC={np.mean(valid_GOC):.4f}, "
                f"GUC={np.mean(valid_GUC):.4f}, "
                f"GTC={np.mean(valid_GTC):.4f}"
            )

            output_final = pred_bin.unsqueeze(1).float()
            target_final = label_bin.unsqueeze(1).float()

            if train:
                save_image(target_final, "Target_train", val_batch_size)
                save_image(output_final, "Prediction_train", val_batch_size)
            else:
                save_image(target_final, "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img tif files")
    parser.add_argument("--val_path", type=str, help="path to img tif files")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_contour",
        help="select distance transform type - dist_mask,dist_contour,dist_contour_tif",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--use_deep_supervision", type=str2bool, default=True, help="Enable deep supervision."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--test_path", type=str, help="path to img tif files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_deep_supervision", type=str2bool, default=True, help="Enable deep supervision."
    )

    return parser























