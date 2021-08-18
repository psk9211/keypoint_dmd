#!/usr/bin/env python
import os
import sys 
import shutil

if len(sys.argv) < 2:
    print('usage:] check.py TARGET_DIR')
    exit(1)
target_dir = sys.argv[1]
print('processing %s' % target_dir)
for f in os.listdir(target_dir):
    if ':' not in f: continue
    new_file_name = f.replace(':', '_')
    print('moving %s to %s' % (f, new_file_name))
    shutil.move(os.path.join(target_dir, f), os.path.join(target_dir, new_file_name))