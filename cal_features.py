from utils.cal_plm_features import *
from utils.data import fasta_to_dict
import time


file_name = r'./data/amp/training_dataset.fasta'
print("Script start")


fasta_dict = fasta_to_dict(file_name)

time1 = time.time()

dict_unirep = cal_UniRep(fasta_dict=fasta_dict, save_path=file_name.replace('.fasta', '_unirep.pkl'))
time2 = time.time()
print('unirep time:', time2 - time1, flush=True)

dict_esm2 = cal_ESM2(fasta_dict=fasta_dict, save_path=file_name.replace('.fasta', '_esm2.pkl'), batch_size=1)
time3 = time.time()
print('esm2 time:', time3 - time2, flush=True)

dict_prott5 = cal_ProtT5(fasta_dict=fasta_dict, save_path=file_name.replace('.fasta', '_prott5.pkl'))
time4 = time.time()
print('prott5 time:', time4 - time3, flush=True)

dict_esmc = cal_ESMC(fasta_dict=fasta_dict, save_path=file_name.replace('.fasta', '_esmc.pkl'))
time5 = time.time()
print('esmc time:', time5 - time4, flush=True)


print('unirep', time2 - time1, flush=True)
print('esm2', time3 - time2, flush=True)
print('prott5', time4 - time3, flush=True)
print('esmc', time5 - time4, flush=True)