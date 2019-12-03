import argparse
import os
import sys
import traceback as tb

from imports.predict import predict

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('jobdirs', help='Directory with trained jobs to load and predict')
    args.add_argument('image_dir', help='Directory with images to predict')
    args.add_argument('-f', type=float, help='Fracture of images that will be used as input', default=1)
    parsed_args = args.parse_args(sys.argv[1:])

    for dir_ in os.listdir(parsed_args.jobdirs):
        path = os.path.join(parsed_args.jobdirs, dir_)
        if not path:
            print(dir_, "is not a dir")
            continue
        print('Dir -', path)
        try:
            predict(path, parsed_args.image_dir, parsed_args.f)
        except Exception as e:
            tb.print_exc()
