import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import os
from torch.utils.data import DataLoader
from dataset import *
import glob
from models import PFAMNet
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser, get_pred_mask
from torch import nn

def build_model(model_type, use_deep_supervision=True):
    if model_type == 'field':
        model = PFAMNet(use_deep_supervision=use_deep_supervision)
    return model
if __name__ == '__main__':
    args = create_validation_arg_parser().parse_args()
    args.model_file = '/workspace/Experiment/pfamnet_pfe_amff_kedaxunfei/best_model.pt'
    args.save_path = '/workspace/Experiment/pfamnet_pfe_amff_kedaxunfei/pred'
    args.model_type = 'field'
    args.test_path = '/workspace/Dataset/kedaxunfei/test/image'
    test_path = os.path.join(args.test_path, '*.tif')
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type
    cuda_no = args.cuda_no
    CUDA_SELECT = 'cuda:{}'.format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else 'cpu')
    test_file_names = glob.glob(test_path)
    test_file_names = [os.path.splitext(filePath)[0] for filePath in test_file_names]
    valLoader = DataLoader(DatasetField(args.test_path, test_file_names, args.model_type))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = build_model(model_type, args.use_deep_supervision)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    for i, data in enumerate(tqdm(valLoader)):
        inputs = data[1].to(device)
        img_file_name = data[0]
        outputs = model(inputs)
        pred_bin = get_pred_mask(outputs[0])
        pred_np = pred_bin.cpu().numpy().astype(np.uint8)
        pred_np = np.squeeze(pred_np)
        output_path = os.path.join(save_path, os.path.basename(img_file_name[0] + '.tif'))
        cv2.imwrite(output_path, pred_np)
import os
import csv
from glob import glob
from tqdm import tqdm
import rasterio
import numpy as np
from skimage.measure import label, regionprops
PRED_DIR = save_path
GT_DIR = args.test_path.replace('image', 'mask')
OUT_CSV = save_path + '/' + 'eval.csv'
THRESH = 0.5

def read_raster_as_binary(path, thresh=0.5, prob=True):
    with rasterio.open(path) as src:
        arr = src.read(1)
    if prob:
        mask = (arr > thresh).astype(np.uint8)
    else:
        mask = (arr > 0).astype(np.uint8)
    return mask

def compute_pixel_metrics(gt, pred):
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    TP = int(np.sum((pred == 1) & (gt == 1)))
    TN = int(np.sum((pred == 0) & (gt == 0)))
    FP = int(np.sum((pred == 1) & (gt == 0)))
    FN = int(np.sum((pred == 0) & (gt == 1)))
    total = TP + TN + FP + FN + 1e-12
    OA = (TP + TN) / total * 100.0
    prec = TP / (TP + FP + 1e-12)
    rec = TP / (TP + FN + 1e-12)
    Precision = prec * 100.0
    Recall = rec * 100.0
    if Precision + Recall > 0:
        F1 = 2 * Precision * Recall / (Precision + Recall)
    else:
        F1 = 0.0
    IOU = TP / (TP + FP + FN + 1e-12) * 100.0
    return {'OA': OA, 'Precision': Precision, 'Recall': Recall, 'F1': F1, 'IoU': IOU, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def label_objects(mask):
    return label(mask, connectivity=1)

def compute_obj_metrics(pred_mask, gt_mask, verbose=False):
    labeled_pred = label_objects(pred_mask)
    labeled_gt = label_objects(gt_mask)
    props_pred = regionprops(labeled_pred)
    props_gt = regionprops(labeled_gt)
    OC, UC, TC, area_list = ([], [], [], [])
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
            oc, uc, tc = (1.0, 1.0, 1.0)
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
    if verbose:
        print(f'GOC={GOC:.3f}, GUC={GUC:.3f}, GTC={GTC:.3f}')
    return (GOC, GUC, GTC, {'OC': OC, 'UC': UC, 'TC': TC})

def main(pred_dir, gt_dir, out_csv, thresh=0.5):
    pred_list = sorted([os.path.basename(p) for p in glob(os.path.join(pred_dir, '*'))])
    gt_list = sorted([os.path.basename(p) for p in glob(os.path.join(gt_dir, '*'))])
    names = sorted(list(set(pred_list).intersection(set(gt_list))))
    if len(names) == 0:
        print('No matching files between predict and mask folders.')
        return
    print(f'Found {len(names)} matched files. Evaluating ...')
    rows = []
    global_TP = global_TN = global_FP = global_FN = 0
    goc_list = []
    guc_list = []
    gtc_list = []
    for name in tqdm(names):
        pred_path = os.path.join(pred_dir, name)
        gt_path = os.path.join(gt_dir, name)
        try:
            pred = read_raster_as_binary(pred_path, thresh=thresh, prob=True)
            gt = read_raster_as_binary(gt_path, thresh=0.5, prob=False)
        except Exception as e:
            print(f'Error reading {name}: {e}, skipping.')
            continue
        if pred.shape != gt.shape:
            try:
                import cv2
                pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred = (pred_resized > 0).astype(np.uint8)
            except Exception as e:
                print(f'Size mismatch {name} and resize failed: {e}. skip.')
                continue
        pix_metrics = compute_pixel_metrics(gt, pred)
        global_TP += pix_metrics['TP']
        global_TN += pix_metrics['TN']
        global_FP += pix_metrics['FP']
        global_FN += pix_metrics['FN']
        GOC, GUC, GTC, _ = compute_obj_metrics(pred, gt, verbose=False)
        goc_list.append(GOC)
        guc_list.append(GUC)
        gtc_list.append(GTC)
        row = {'name': name, 'OA': pix_metrics['OA'], 'Precision': pix_metrics['Precision'], 'Recall': pix_metrics['Recall'], 'F1': pix_metrics['F1'], 'IoU': pix_metrics['IoU'], 'GOC': GOC, 'GUC': GUC, 'GTC': GTC, 'TP': pix_metrics['TP'], 'TN': pix_metrics['TN'], 'FP': pix_metrics['FP'], 'FN': pix_metrics['FN']}
        rows.append(row)
    fieldnames = ['name', 'OA', 'Precision', 'Recall', 'F1', 'IoU', 'GOC', 'GUC', 'GTC', 'TP', 'TN', 'FP', 'FN']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    import math
    n = len(rows) if len(rows) > 0 else 1
    mean_OA = sum([r['OA'] for r in rows]) / n
    mean_Precision = sum([r['Precision'] for r in rows]) / n
    mean_Recall = sum([r['Recall'] for r in rows]) / n
    mean_F1 = sum([r['F1'] for r in rows]) / n
    mean_IoU = sum([r['IoU'] for r in rows]) / n
    mean_GOC = sum(goc_list) / n if len(goc_list) > 0 else float('nan')
    mean_GUC = sum(guc_list) / n if len(guc_list) > 0 else float('nan')
    mean_GTC = sum(gtc_list) / n if len(gtc_list) > 0 else float('nan')
    TP = global_TP
    TN = global_TN
    FP = global_FP
    FN = global_FN
    total = TP + TN + FP + FN + 1e-12
    global_OA = (TP + TN) / total * 100.0
    global_Prec = TP / (TP + FP + 1e-12) * 100.0
    global_Rec = TP / (TP + FN + 1e-12) * 100.0
    global_IoU = TP / (TP + FP + FN + 1e-12) * 100.0
    if global_Prec + global_Rec > 0:
        global_F1 = 2 * global_Prec * global_Rec / (global_Prec + global_Rec)
    else:
        global_F1 = 0.0
    print('=' * 30)
    print(f'Evaluated {n} images. Results saved to: {out_csv}')
    print('Per-image mean (arithmetic):')
    print(f' OA={mean_OA:.4f}  Precision={mean_Precision:.4f}  Recall={mean_Recall:.4f}  F1={mean_F1:.4f}  IoU={mean_IoU:.4f}')
    print(f' Obj-level mean: GOC={mean_GOC:.4f}  GUC={mean_GUC:.4f}  GTC={mean_GTC:.4f}')
    print('Global (pixel-aggregated) metrics:')
    print(f' OA={global_OA:.4f}  Precision={global_Prec:.4f}  Recall={global_Rec:.4f}  F1={global_F1:.4f}  IoU={global_IoU:.4f}')
    print('=' * 30)
if __name__ == '__main__':
    main(PRED_DIR, GT_DIR, OUT_CSV, thresh=THRESH)
