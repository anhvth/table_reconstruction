import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--link', required=True, help='Format: https://drive.google.com/open?id=fid')
args = parser.parse_args()
fid = args.link.split('=')[-1]
os.system('~/gdrive-linux-x64 download {}'.format(fid))
