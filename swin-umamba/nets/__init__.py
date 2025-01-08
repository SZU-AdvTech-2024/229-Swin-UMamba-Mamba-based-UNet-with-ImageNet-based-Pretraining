from .vmunet import load_vm_model,load_rm_model,load_mhs_vm_model,load_vm1_model
from .vmunet_v2 import load_vm2_model,load_vm3_model
from .Swin_umamba import getmodel
from .NewSwinUM import load_new_model,load_SwinPA_UM
from .SwinUMambaD import get_swin_umambaD
from .MedMamba import MedMamba,medmamba_t
from .CMUNet import CMUNet

def net(model_name):
    if model_name == 'VMUNet':
        model = load_vm_model()
    elif model_name == 'VMUNetv1':
        model = load_vm1_model()
    elif model_name == 'VMUNetv2':
        model = load_vm2_model()
    elif model_name =='RMUNet':
        model = load_rm_model()
    elif model_name == 'SwinUMamba':
        model = getmodel()
    elif model_name == 'SwinUMambaD':
        model = get_swin_umambaD()
    elif model_name == 'NewSwinUM':
        model = load_SwinPA_UM()
    elif model_name == 'MHS_VMUNet':
        model = load_mhs_vm_model().cuda()
    elif model_name == 'MHS_VMUNetv1':
        model = load_mhs_vm_model().cuda()
    elif model_name == "NewSwinUM2":
        model = load_new_model().cuda()
    elif model_name == "CMUNet":
        model = CMUNet().cuda()
    else:
        print("No model!")
        return
    return model

# Thyroid_path = "/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/"
# KvasirSeg_path = "./data/Kvasir-Seg/"
# KvasirIns_path = "./data/Kvasir-Instrument/"
# CVC_Clinic_path = "./data/CVC-ClinicDB/"
# CVC_Colon_path = "./data/CVC-ColonDB/"
# CVC_300_path = "./data/CVC-300/"

def get_dataset(datasets):
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    elif datasets == 'kvasir-ins':
        data_path = './data/Kvasir_Instrument/'
    elif datasets == 'kvasir':
        data_path = './data/Kvasir-Seg/'
    elif datasets == 'cvc-300':
        data_path = './data/CVC-300/'
    elif datasets == 'cvc-Colon':
        data_path = './data/CVC-ColonDB/'
    elif datasets == 'cvc-Clinic':
        data_path = './data/CVC-ClinicDB/'
    elif datasets == 'polpy':
        data_path = './data/Polpy/'
    elif datasets == 'Thyroid':
        data_path = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/'
    elif datasets == 'DDTI' :
        data_path = './data/DDTI/'
    elif datasets == 'Breast':
        data_path = './data/Breast/'
    elif datasets == 'BUSI':
        data_path = './data/BUSI/'
    else:
        raise Exception('datasets in not right!')
    return data_path


