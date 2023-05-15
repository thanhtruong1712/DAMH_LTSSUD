from XGBoost import load_model, predict_proba
from xgboost import XGBClassifier
from argparse import ArgumentParser
from numpy import loadtxt
from time import time
from XGB_par_v0 import predict_proba_kernel
import numpy as np

def create_argument_parser():
    '''
    Hàm phân tích cú pháp tham số dòng lệnh
    '''
    parser = ArgumentParser()

    parser.add_argument('X_test', help='đường dẫn tới tập dữ liệu kiểm tra')
    parser.add_argument('learner_path', help='đường dẫn tập tin các siêu tham số của mô hình đã lưu')
    parser.add_argument('tree_path', help='đường dẫn tập tin cấu trúc các cây thành phần của mô hình đã lưu')

    parser.add_argument('-p', '--parallel', type=int, choices=[0, 1, 2, 3], default=0,
                                        help='''chạy tuần tự hoặc song song:
                                                    0 - chạy tuần tự;
                                                    1 - chạy song song;
                                                    2 - chạy song song và tối ưu hóa (v1);
                                                    3 - chạy song song và tối ưu hóa (v2);''')
    parser.add_argument('-b', '--blockSize', type=int, default=0)
    
    return parser

if __name__ == '__main__':
    # đọc tham số dòng lệnh
    parser = create_argument_parser()
    args = parser.parse_args()

    # xác định số cột đặc trưng, bỏ qua cột tên ảnh ở cuối
    with open(f'{args.X_test}') as f:
        ncols = len(f.readline().split(','))

    X_test = loadtxt(f'{args.X_test}',delimiter=',', skiprows=1, usecols=range(ncols-1))

    # đọc mô hình và dự đoán - tự cài đặt
    print('Self implement loader')
    start = time()
    n_estimators, n_classes, features, values = load_model(args.learner_path, args.tree_path)
    print(f'model loaded in {(time() - start)*1000:f}ms')

    start = time()
    if args.parallel == 0:
        xgb_loader_pred = predict_proba(X_test, features, values, n_classes)
    elif args.parallel == 1:
        xgb_loader_pred = predict_proba_kernel(X_test, features, values, n_classes, args.blockSize)
        
    xgb_loader_pred = np.argmax(xgb_loader_pred, 1)

    print(f'prediction in {(time() - start)*1000:f}ms')

    # đọc mô hình và dự đoán - thư viện xgboost
    xgblib_model = XGBClassifier(objective='multi:softmax', use_label_encoder=False,
                                                    num_class=n_classes, n_estimators=n_estimators, learning_rate=0.2)
    xgblib_model.load_model(args.learner_path)    
    xgblib_pred = np.argmax(xgblib_model.predict_proba(X_test), 1)
    
    print(xgb_loader_pred)
    print(xgblib_pred)

    print(f'error with xgboost library classifier: {np.count_nonzero(xgb_loader_pred - xgblib_pred)}')
