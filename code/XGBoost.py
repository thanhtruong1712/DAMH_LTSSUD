from numba import njit
from numpy import array, empty_like, empty, int32, float64, zeros
from math import exp
from queue import Queue
import json

@njit
def softmax(arr:array):
    result = empty_like(arr)
    nrows, ncols = arr.shape

    for i in range(nrows):
        row_sum = 0
        for j in range(ncols):
            element_exp = exp(arr[i, j])
            result[i, j] = element_exp
            row_sum += element_exp

        for j in range(ncols):
            result[i, j] /= row_sum

    return result


def load_tree(feature_arr:array, value_arr:array, tree_dict:dict):
    '''
    Hàm đọc cấu trúc cây ở dạng từ điển.
    Kết quả đọc được lưu vào mảng đặc trưng và giá trị truyền vào.

    Đầu vào:
    - feature_arr, value_arr (numpy.array): mảng lưu đặc trưng và giá trị của các nút trong cây
    - tree_dict (dict): thông tin về cây quyết định lưu ở dạng từ điển
    '''
    # tạo hàng đợi FIFO các nút và chỉ mục tương ứng trong mảng
    tree_q = Queue()
    tree_q.put(tree_dict)

    idx_q = Queue()
    idx_q.put(0)

    while not tree_q.empty():
        node = tree_q.get()
        idx = idx_q.get()

        # xác định nút lá hay nút nhánh
        # nếu có khóa 'leaf' là nút lá
        node_val = node.get('leaf')
        if node_val is None:
            # lưu đặc trưng và giá trị ngưỡng phân nhánh
            feature_arr[idx] = int32(node['split'][1:])
            value_arr[idx] = float64(node['split_condition'])

            # thêm vào hàng đợi thông tin và chỉ mục trong mảng của nút con
            tree_q.put(node['children'][0])
            tree_q.put(node['children'][1])

            idx_q.put(2 * idx + 1)
            idx_q.put(2 * idx + 2)
        else:
            feature_arr[idx] = -1 # giá trị đặc biệt thể hiện nút lá
            value_arr[idx] = float64(node_val)


def load_model(hyperparams_path:str, tree_path:str):
    '''
    Hàm đọc mô hình XGBoost đã lưu.
    Mỗi cây trong mô hình được lưu thành mảng 1 chiều bao gồm 2 mảng đặc trưng và giá trị.
    Mảng giá trị thê hiện giá trị phân nhánh hoặc nút lá tùy giá trị đặc trưng tại ô tương ứng.

    Đầu vào:
    - hyperparams_path, tree_path (str): lần lượt là đường dẫn tới tập tin lưu các siêu tham số
    và lưu cấu trúc cây của mô hình.

    Đầu ra:
    - n_estimators (int): số (nhóm) cây thành phần của mô hình.
    - n_classes (int): số phân lớp đã huấn luyện
    - features (numpy.array): mảng thể hiện đặc trưng được xét tại nút trong cây.
    - values (numpy.array): mảng thể hiện giá trị phân nhánh hoặc nút lá.
    '''
    # mở tập tin, phân tích cú pháp và lưu thông tin vào từ điển
    with open(hyperparams_path) as f:
        hyperparams = json.load(f)

    with open(tree_path) as f:
        tree_list = json.load(f)

    # truy xuất các siêu tham số
    attributes = json.loads(hyperparams['learner']['attributes']['scikit_learn'])
    n_estimators = attributes['n_estimators']
    n_classes = attributes['n_classes_']
    max_depth = attributes['max_depth']
    if max_depth is None:
        max_depth = 6

    # xác định số nút tối đa của cây thành phần và tạo mảng.
    nrows = len(tree_list)
    ncols = sum((2**i for i in range(max_depth+1)))
    features = empty((nrows, ncols), int32)
    values = empty((nrows, ncols), float64)

    # đọc và nạp thông tin các cây vào mảng
    for feat_arr, val_arr, tree in zip(features, values, tree_list):
        load_tree(feat_arr, val_arr, tree)

    return n_estimators, n_classes, features, values

@njit
def traverse_tree(x:array, features_arr:array, values_arr:array):
    '''
    Hàm duyệt cây quyết định và đưa ra dự đoán chon một đối tượng.

    Đầu vào:
    - x (numpy.array): đối tượng dự đoán.
    - features_arr, values_arr (numpy.array): mảng đặc trưng và giá trị của cây quyết định.

    Đầu ra:
    - value (float): giá trị dự đoán.
    '''
    idx = 0

    # kiểm tra nút lá
    while features_arr[idx] != -1:
        # so sánh giá trị đặc trưng của đối tượng và giá trị phân nhánh
        # duyệt tới nút con phù hợp.
        if x[features_arr[idx]] - values_arr[idx] < 0:
            idx = 2 * idx + 1
        else:
            idx = 2 * idx + 2

    return values_arr[idx]

@njit
def predict_proba(X:array, features_arr:array, values_arr:array, n_classes:int32):
    '''
    Hàm dự đoán tập dữ liệu trên mô hình XGBoost được đọc.

    Đầu vào:
    - X (numpy.array): tập dữ liệu dự đoán.
    - features_arr, values_arr (numpy.array): mảng đặc trưng và giá trị của cây quyết định.
    - n_classes (int): số phân lớp được huấn luyện.

    Đầu ra:
    - y_pred (numpy.array): mảng dự đoán xác suất thuộc về các phân lớp của các đối tượng
    '''
    y_pred = zeros((X.shape[0], n_classes))

    for i, x in enumerate(X):
        for j, (f_arr, v_arr) in enumerate(zip(features_arr, values_arr)):
            y_pred[i, j % n_classes] += traverse_tree(x, f_arr, v_arr)

    return softmax(y_pred)