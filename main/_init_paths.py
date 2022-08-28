import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

this_dir = os.path.dirname(__file__)
#print('@current dir (ref): %s'%this_dir)
add_path(os.path.join(this_dir))
add_path(os.path.join(this_dir, '..'))
add_path(os.path.join(this_dir, '..', 'models'))
add_path(os.path.join(this_dir, '..', 'utils'))

#print("=================SYS PATH================\n")
#for path in sys.path:
    #print(path)
#print("\n=================SYS PATH================")
