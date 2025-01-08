import argparse

Thyroid_path = "/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/"
KvasirSeg_path = "./data/Kvasir-Seg/"
KvasirIns_path = "./data/Kvasir-Instrument/"
CVC_Clinic_path = "./data/CVC-ClinicDB/"
CVC_Colon_path = "./data/CVC-ColonDB/"
CVC_300_path = "./data/CVC-300/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DDTI",type=str,
                        help='dataset name')
    parser.add_argument('--root_dir', default="./data/DDTI/", type=str,
                        help='dataset root path')
    parser.add_argument('--model_pth', default="DDTI_best", type=str,
                        help='dataset root path')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=24, type=int, metavar='N',help='mini-batch size')
    parser.add_argument('--img_size', type=int,
                        default=256, help='input patch size of network input')
    parser.add_argument('--model', default="SwinUMambaD", help='training model')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    args = parser.parse_args()

    return args
