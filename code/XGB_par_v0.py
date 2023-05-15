from numba import cuda
from math import exp, ceil

@cuda.jit
def softmax_kernel(arr, ncols, result):
    '''
    Hàm tính softmax song song theo từng dòng trên mảng 2 chiều
    biểu diễn theo mảng 1 chiều.
    Kết quả lưu vào mảng softmax truyền vào.
    
    Đầu vào:
    - arr (cuda.device_array): mảng giá trị đầu vào và lưu kết quả trung gian theo dạng 1 chiều.
    - ncols (int): số cột của mảng arr
    - result (cuda.device_array): mảng lưu kết quả.
    '''
    # kiểm tra chỉ mục luồng song song nằm trong mảng arr
    i = cuda.grid(1)
    if i < arr.size:
        # tính e^x
        _exp = exp(arr[i])
        arr[i] = _exp

        # xác định giá trị đang tính thuộc dòng nào
        # tính tổng e^x theo dòng
        row_1d = i - (i % ncols)
        cuda.atomic.add(arr, row_1d, _exp)

        cuda.syncthreads()
        
        result[i] = _exp / arr[row_1d]
        
@cuda.jit
def memset_kernel(arr, val):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = val

@cuda.jit
def traverse_tree_kernel(Xt, tree_features, tree_values, result):
    '''
    Hàm duyệt song song các đối tượng trên 1 cây.
    Kết quả lưu vào mảng result
    
    Đầu vào:
    - Xt (cuda.device_array): tập dữ liệu  dự đoán, kích thước (đặc trưng, đối tượng)
    - tree_features, tree_values (cuda.device_array): mảng đặc trưng và giá trị của các cây quyết định.
    - result (cuda.device_array): mảng kết quả.
    '''
    # xác định đối tượng
    i = cuda.grid(1)
    if i < Xt.shape[1]:
        # lấy giá trị đặc trưng tại nút
        idx = 0
        feat = tree_features[idx]
        val = tree_values[idx]
        
        # so sánh giá trị đối tượng
        # di chuyển đến nhánh phù hợp
        while feat != -1:
            if Xt[feat, i] < val:
                idx = 2 * idx + 1
            else:
                idx = 2 * idx + 2
                
            feat = tree_features[idx]
            val = tree_values[idx]
            
        # lưu giá trị nút lá
        # result[i] += val
        cuda.atomic.add(result, i, val)
        
        
def predict_proba_kernel(X, features_arr, values_arr, n_classes, block_size):
    '''
    Hàm dự đoán song song các đối tượng trong tập dữ liệu.
    
    Đầu vào:
    - X (numpy.array): tập dữ liệu dự đoán
    - features_arr, values_arr (cuda.device_array): mảng đặc trưng và giá trị của các cây quyết định.
    - n_classes (int): số phân lớp của mô hình, kết quả từ hàm XGBoost.load_model
    - block_size (int): số luồng chạy song song.
    
    Đầu ra:
    - result_final (numpy.array): mảng dự đoán xác suất các phân lớp của các đối tượng.
    '''
    # tạo mảng lưu kết quả trong device
    d_result = cuda.device_array((n_classes, X.shape[0]))
    
    # sao chép mô hình vào device
    d_Xt = cuda.to_device(X.T)
    d_features = cuda.to_device(features_arr)
    d_values = cuda.to_device(values_arr)

    # tính grid_size cho hàm khởi tạo giá trị và duyệt cây
    grid_size = ceil(X.shape[0]  / block_size)

    # khởi tạo giá trị mảng kết quả
    for arr in d_result:
        memset_kernel[grid_size, block_size](arr, 0)

    cuda.synchronize()

    # duyệt cây
    for i, (f_arr, v_arr) in enumerate(zip(d_features, d_values)):
        traverse_tree_kernel[grid_size, block_size](d_Xt, f_arr, v_arr, d_result[i % n_classes])
        
    cuda.synchronize()
    
    # thay đổi kích thước và tính softmax
    d_result = d_result.T.reshape(-1)
    d_result_final = cuda.device_array_like(d_result)
    grid_size = ceil(d_result.size / block_size)
    softmax_kernel[grid_size, block_size](d_result, n_classes, d_result_final)
    
    return d_result_final.reshape((X.shape[0], n_classes)).copy_to_host()
