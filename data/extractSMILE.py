import os
from glob import glob
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('-audio_folder', dest='audio_folder', default='wav/', type=str)
parser.add_argument('-features_folder', dest='features_folder', default='extracted_features/', type=str)
args = parser.parse_args()
    
# Modify openSMILE paths HERE:
SMILEpath = '../../../opt/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
SMILEconf = '../..../opt/opensmile-2.3.0/config/emobase_v2.conf'


# Paths
audio_folders = glob(args.audio_folder)
features_folder = args.features_folder

def extract_iemocap(audio_folder, features_folder):
    # Load file list
    instances = glob(audio_folder+'wav_mod/*')
    all_files = []
    for i in instances:
        all_files += glob(i+'/*.wav')
    # Iterate through partitions and extract features
    if not os.path.exists(features_folder):
        os.mkdir(features_folder)       
    # Extract openSMILE features for the whole partition (LLD-only)
    for f in all_files:
        os.system(SMILEpath + ' -C ' + SMILEconf + ' -I ' + f + \
        ' -O ' + features_folder + f.split("/")[-1].replace('wav','arff'))

for path in audio_folders:
    extract_iemocap(path, features_folder)
