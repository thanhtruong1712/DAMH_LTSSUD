from XGBoost import load_model, predict_proba
from xgboost import XGBClassifier
from argparse import ArgumentParser
from time import time
from glob import glob
from tqdm import tqdm
import numpy as np
import warnings
from feature_extract import getFigureForImage2, getFigureForImage

from feature_extract_v0 import getFigureForImage3
from XGB_par_v0 import predict_proba_kernel

from feature_extract_v1 import getFigureForImage4
from XGB_par_v1 import predict_proba_kernel1


def create_argument_parser():
    '''
    Hàm phân tích cú pháp tham số dòng lệnh
    '''
    parser = ArgumentParser()

    parser.add_argument('learner_path', help='đường dẫn tập tin các siêu tham số của mô hình đã lưu')
    parser.add_argument('tree_path', help='đường dẫn tập tin cấu trúc các cây thành phần của mô hình đã lưu')

    parser.add_argument('-d', help='đường dẫn tới thư mục hình đầu vào')
    parser.add_argument('-i', '--image', help='đường dẫn tới hình đầu vào')
    parser.add_argument('--npy', help='đường dẫn tới tập tin đặc trưng đã trích xuất')
    parser.add_argument('-p', '--parallel_version', help='''lựa chọn phiên bản chạy phiên bản song song:
                                                                                    0: song song
                                                                                    1: song song, tối ưu (v1)
                                                                                    2: song song, tối ưu (v2)''',
                                        type=int, choices=[0,1,2], default=-1)
    parser.add_argument('-b', '--blockSize', type=int, default=32)
    
    return parser


def get_sort_key(s):
    start = s.rfind('_') + 1
    end = s.rfind('.')
    return int(s[start:end])

def run_library(args, n_estimators, n_classes):
    '''
    Hàm thực hiện rút trích đặc trưng và dự đoán 1 hoặc 1 tập ảnh đầu vào sử dụng thư viện opencv và xgboost.
    Trả về bộ kết quả rút trích và dự đoán.
    '''
    print('Baseline: opencv & xgboost library')

    if args.d is not None:
        test_ls = glob(f'{args.d}/Test_*')
        test_ls = sorted(test_ls, key=get_sort_key)
    
        # rút trích đặc trưng - sử dụng thư viện
        X_test = [getFigureForImage(path) for path in tqdm(test_ls, 'Feature extraction')]
        cv_X_test = np.vstack(X_test)
    elif args.image is not None:
        start = time()
        cv_X_test = getFigureForImage(args.image)
        print(f'Feature extraction: {(time() - start) * 1000}ms')

        cv_X_test = np.atleast_2d(cv_X_test)

    xgblib_model = XGBClassifier(objective='multi:softmax', use_label_encoder=False,
                                                    num_class=n_classes, n_estimators=n_estimators, learning_rate=0.2)

    xgblib_model.load_model(args.learner_path)

    start = time()
    xgblib_pred = xgblib_model.predict(cv_X_test)
    print(f'XGBoost prediction: {(time() - start)*1000:f}ms', '\n')

    return cv_X_test, xgblib_pred

def run_sequential(args, features, values, n_classes):
    print('Sequential implementation')

    if args.d is not None:
        test_ls = glob(f'{args.d}/Test_*')
        test_ls = sorted(test_ls, key=get_sort_key)

        # rút trích đặc trưng - tự cài đặt
        image_features = [getFigureForImage2(path) for path in tqdm(test_ls, 'Feature extraction')]
        X_test = np.vstack(image_features)
    elif args.image is not None:
        start = time()
        X_test = getFigureForImage2(args.image)
        print(f'Feature extraction: {(time() - start) * 1000}ms')

        X_test = np.atleast_2d(X_test)

    start = time()
    xgb_pred = np.argmax(predict_proba(X_test, features, values, n_classes), 1)
    print(f'XGBoost prediction: {(time() - start)*1000:f}ms', '\n')

    return X_test, xgb_pred

def run_parallel(args, features, values, n_classes):
    print('Parallel implementation ', end='')

    if args.parallel_version == 0:
        print('v0')
    elif args.parallel_version == 1:
        print('v1')
    elif args.parallel_version == 2:
        print('v2')

    if args.d is not None:
        test_ls = glob(f'{args.d}/Test_*')
        test_ls = sorted(test_ls, key=get_sort_key)

        # rút trích đặc trưng - tự cài đặt
        if args.parallel_version != 2:
            image_features = [getFigureForImage3(path) for path in tqdm(test_ls, 'Feature extraction')]
        else:
            image_features = [getFigureForImage4(path) for path in tqdm(test_ls, 'Feature extraction')]

        X_test = np.vstack(image_features)
    elif args.image is not None:
        start = time()
        if args.parallel_version != 2:
            X_test = getFigureForImage3(args.image)
        else:
            X_test = getFigureForImage4(args.image)
        print(f'Feature extraction: {(time() - start) * 1000}ms')

        X_test = np.atleast_2d(X_test)

    start = time()
    xgb_pred = np.argmax(predict_proba_kernel(X_test, features, values, n_classes, args.blockSize), 1)
    print(f'XGBoost prediction v0: {(time() - start)*1000:f}ms')
        
    if args.parallel_version > 0:
        start = time()
        xgb_pred = np.argmax(predict_proba_kernel1(X_test, features, values, n_classes, args.blockSize), 1)
        print(f'XGBoost prediction v1: {(time() - start)*1000:f}ms')
    
    return X_test, xgb_pred

def compare(features, predictions):
    print(f'\nfeatures mean error: {np.abs(features[0] - features[1]).mean()}')
    print(f'error with xgboost library classifier: {np.count_nonzero(predictions[0] != predictions[1])}')


if __name__ == '__main__':
    # đọc tham số dòng lệnh
    parser = create_argument_parser()
    args = parser.parse_args()

    n_estimators, n_classes, features, values = load_model(args.learner_path, args.tree_path)

    if args.parallel_version == -1:
        base_X_test, xgb_base_pred = run_library(args, n_estimators, n_classes)
        X_test, xgb_pred = run_sequential(args, features, values, n_classes)
        np.savez('sequential.npz', seq_X_test=X_test, xgb_seq_pred=xgb_pred)
    else:
        base_X_test, xgb_base_pred = run_sequential(args, features, values, n_classes)

        # with np.load('sequential.npz') as features_n_pred:
        #     base_X_test = features_n_pred['seq_X_test']
        #     xgb_base_pred = features_n_pred['xgb_seq_pred']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_test, xgb_pred = run_parallel(args, features, values, n_classes)

    compare((base_X_test, X_test), (xgb_base_pred, xgb_pred))
