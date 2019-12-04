import os
import sys
import traceback as tb
from imports.train_test import train_test
from multiprocessing import Process

if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_test()
    settings = sys.argv[1]
    if os.path.isfile(settings):
        print('File -', settings)
        train_test(settings)
    elif os.path.isdir(settings):
        print('Dir -', settings)
        for f in os.listdir(settings):
            print('File -', os.path.join(settings, f))
            try:
                proc = Process(target=train_test, args=[os.path.join(settings, f)])
                proc.start()
                proc.join()
            except Exception as e:
                tb.print_exc()
