from glob import glob 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import os
import numpy as np
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('-features_folder', dest='features_folder', default='extracted_features/', type=str)
args = parser.parse_args()

def getFeature(File_path):
    File = open(File_path,"r")
    Data = File.readlines()
    
    Mfcc = np.arange(4,16)
    Mfcc_de = np.arange(30,42)
    Mfcc_de_de = np.arange(56,68)
    
    Lound = np.array([3])
    Lound_de = np.array([29])
    Lound_de_de = np.array([55])
    
    F0 = np.array([26])
    F0_de = np.array([52])
    
    Voice_Pro = np.array([25])
    Voice_Pro_de = np.array([51])
    
    Zcr = np.array([24])
    Zcr_de = np.array([50])
    
    Index = np.concatenate((F0,F0_de,Lound,Lound_de,Lound_de_de,Voice_Pro,Voice_Pro_de,Zcr,Zcr_de,Mfcc,Mfcc_de,Mfcc_de_de))
    
    All_Feature = []
    for data in Data[86:]:
        feature = np.array(np.array(data.split(","))[Index],dtype=float)
        All_Feature.append(feature)
        
    All_Feature = np.asarray(All_Feature)
    
    return All_Feature

if __name__ == '__main__':
    features_folder = args.features_folder
    all_feat = []
    feat_len = []
    feat_name = []
    # initialize dictionary
    speakers = {}
    for s in range(1, 6):
        for g in ["F", "M"]:
            speakers["Ses0{}{}".format(s, g)] = []

    all_files = glob(features_folder+"/*.arff")
    for f in all_files:
        speakers[f.split("/")[-1][:6]].append(f)

    # speaker normalization
    all_feat = {}
    for spk in speakers:
        print("processing speaker {} ...".format(spk))
        name = []
        feat = []
        length = []
        for f in speakers[spk]:
            try:
                feat_ = getFeature(f)
            except:
                print("DATA MISSING!!!! Speaker: {}".format(spk), 'Path: {}'.format(f))
                continue
            #idx_F0 = np.where(feat_[:,0]==0)[0]
            #feat_ = np.delete(feat_ ,idx_F0 ,axis=0)
            try:
                feat = np.concatenate([feat, feat_])
            except:
                feat = feat_
            print(feat_.shape)
            length.append(feat_.shape[0])
            name.append(f.split("/")[-1].split(".")[0])
        feat_zscore = stats.zscore(feat)
        print("Speaker: {}, total utt: {}".format(spk, len(name)))
        idx = 0
        for n, l in zip(name, length):
            all_feat[n] = feat_zscore[idx:idx+l]
            idx += l
    all_feat["pad"] = np.zeros(45)

    print("save as .pkl")
    joblib.dump(all_feat, "feat_all.pkl")            
