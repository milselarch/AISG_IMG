import os

import main

if __name__ == '__main__':
    main.main(
        f'{os.getcwd()}/../local_test/test_input',
        '../local_test/test_output/submission.csv'
    )