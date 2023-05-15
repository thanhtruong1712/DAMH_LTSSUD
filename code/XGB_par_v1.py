from numba import cuda, float64
from math import ceil
from XGB_par_v0 import memset_kernel, traverse_tree_kernel
from itertools import cycle

@cuda.jit(device=True)
def memset_device_func(arr, idx, val):
    arr[idx] = val

@cuda.jit
def softmax_kernel(arr, ncols, result):
    nrows_blk = int(cuda.blockDim.x / ncols)
    rows_sum = cuda.shared.array(0, float64)

    if cuda.threadIdx.x < nrows_blk:
        memset_device_func(rows_sum, cuda.threadIdx.x, 0)

    i = cuda.grid(1)
    if i < arr.size:
        _exp = cuda.libdevice.exp(arr[i])

        row = cuda.threadIdx.x // ncols
        cuda.atomic.add(rows_sum, row, _exp)
        
        cuda.syncthreads()

        result[i] = _exp / rows_sum[row]
        

def predict_proba_kernel1(X, features_arr, values_arr, n_classes, block_size, n_streams=32):
    cuda_streams = cycle([cuda.stream() for _ in range(n_streams)])

    d_result = cuda.device_array((n_classes, X.shape[0]))

    d_Xt = cuda.to_device(X.T, cuda_streams.__next__())
    d_features = cuda.to_device(features_arr, cuda_streams.__next__())
    d_values = cuda.to_device(values_arr, cuda_streams.__next__())

    grid_size = ceil(X.shape[0]  / block_size)

    # for arr, stream in zip(d_result, cuda_streams):
    #     memset_kernel[grid_size, block_size, stream](arr, 0)
    for arr in d_result:
        memset_kernel[grid_size, block_size, cuda_streams.__next__()](arr, 0)

    cuda.synchronize()

    for i, (f_arr, v_arr) in enumerate(zip(d_features, d_values)):
        # traverse_tree_kernel[grid_size, block_size, cuda_streams[i % n_streams]](d_Xt, f_arr, v_arr, d_result[i % n_classes])
        traverse_tree_kernel[grid_size, block_size, cuda_streams.__next__()](d_Xt, f_arr, v_arr, d_result[i % n_classes])

    cuda.synchronize()

    d_result = d_result.T.reshape(-1)
    d_result_final = cuda.device_array_like(d_result)
    grid_size = ceil(d_result.size / block_size)
    softmax_kernel[grid_size, block_size, 0, X.shape[0] * 8](d_result, n_classes, d_result_final)
    
    return d_result_final.reshape((X.shape[0], n_classes)).copy_to_host()
    