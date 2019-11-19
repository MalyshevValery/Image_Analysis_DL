from tests.tools.colors import Colors
import traceback
import sys
import time

class Test(object):
    test_count = 1

    def __init__(self, test_name):
        self.test_number = Test.test_count
        self.test_name = test_name
        Test.test_count += 1

    def __enter__(self):
        self.start = time.time()
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time.time()
        self.test_name += ' Time: {0:.2f}'.format(self.end - self.start)
        if exc_value is None:
            print('{}OK {}: {}{}'.format(Colors.OKGREEN, self.test_number, self.test_name, Colors.ENDC))
        else:
            print('{}FAIL {}: {}{}'.format(Colors.FAIL, self.test_number, self.test_name, Colors.ENDC))
            sys.stdout.flush()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            return False
        return True
