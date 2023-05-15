import cv2 as cv, numpy as np
from numba import cuda
from feature_extract import (convert_rgb2gray, create_gaussian_filter,
                                                applyCannyThreshold, fd_histogram2, fd_hu_moments2, joinNeighborPixel, zipImage)
import math

# phiên bản tự cài dặt
# --------------------------------------------------------------------
@cuda.jit
def apply_kernel_device(image, kernel, out):
    '''
    Hàm thực hiện phép tích chập tại một điểm ảnh với bộ lọc.
    Tại vùng biên, phần tử lân cận gần nhất được chọn làm phần tử đệm.
    Độ dời khi duyệt là 1.
    Chạy song song trên GPU.

    Đầu vào (numba.cuda.cudadrv.devicearray.DeviceNDArray):
    - image: ảnh đầu vào.
    - kernel: bộ lọc.
    - out: mảng lưu kết quả.
    '''
    out_c, out_r = cuda.grid(2)
    offset = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    last_row = image.shape[0] - 1
    last_col = image.shape[1] - 1

    out_pixel = 0.
    if out_r < image.shape[0] and out_c < image.shape[1]:
        for filter_r, r in enumerate(range(out_r - offset[0], out_r + offset[0] + 1)):
            for filter_c, c in enumerate(range(out_c - offset[1], out_c + offset[1] + 1)):
                in_r = min(max(0, r), last_row)
                in_c = min(max(0, c), last_col)
                out_pixel += image[in_r, in_c] * kernel[filter_r, filter_c]

        out[r, c] = np.uint8(math.floor(out_pixel))

def gaussian_blur_kernel(image, ksize, sigmaX, sigmaY=0, block_size=(32, 32)):
    '''
    Hàm làm mờ ảnh sử dụng bộ lọc Gauss.

    Đầu vào:
    - image (np.array): ảnh đầu vào.
    - ksize (tuple): kích thước bộ lọc Gauss, thể hiện số dòng, số cột.
    - sigmaX, sigmaY (float): độ lệch chuẩn cho giá trị bộ lọc theo chiều dọc và ngang.
    - block_size (tuple): kích thước khối chạy song song trên GPU.

    Đầu ra
    - blur (np.array): ảnh kết quả sau khi áp dụng bộ lọc
    '''
    d_kernel = cuda.to_device(create_gaussian_filter(ksize, sigmaX, sigmaY))

    if not cuda.is_cuda_array(image):
        d_image = cuda.to_device(image)
    else:
        d_image = image

    d_out = cuda.device_array(d_image.shape, np.uint8)

    grid_size = (math.ceil(image.shape[1] / block_size[0]),
                        math.ceil(image.shape[0] / block_size[1]))
    apply_kernel_device[grid_size, block_size](d_image, d_kernel, d_out)

    return d_out

# def joinNeighborPixelKernel(src, zip_x, zip_y, mask_size, ratio, block_size):
#     '''
#     Hàm liên kết những khối điểm ảnh lân cận để lấp những điểm ảnh khuyết, chạy song song trên GPU.
#     Kết quả là mảng đánh dấu những điểm ảnh được liên kết và lấp khuyết.

#     Đầu vào:
#     - src (numba.cuda.cudadrv.devicearray.DeviceNDArray): ảnh đầu vào.
#     - zip_x, zip_y (int): kích thước khối theo chiều ngang và dọc.
#     - mask_size (int): kích thước 1 khối lân cận xem xét liên kết.
#     - ratio (float): tỷ lệ nén.
#     - block_size (tuple): kích thước khối chạy song song trên GPU.

#     Đầu ra:
#     - mask_image (numba.cuda.cudadrv.devicearray.DeviceNDArray): mảng đánh dấu những khối ảnh được liên kết.
#     '''
#     zip_rs = math.ceil(src.shape[0] / zip_y)
#     zip_cs = math.ceil(src.shape[1] / zip_x)
#     nonzero_count = cuda.device_array((zip_rs*mask_size, zip_cs*mask_size), np.uint32)

#     if not cuda.is_cuda_array(src):
#         d_src = cuda.to_device(src)
#     else:
#         d_src = src

#     zip_blk_x = zip_x * mask_size
#     zip_blk_y = zip_y * mask_size

#     grid_size = (math.ceil(src.shape[1] / block_size[0]), math.ceil(src.shape[0] / block_size[1]))

#     for start_c in range(0, src.shape[1], zip_x):
#         for start_r in range(0, src.shape[0], zip_y):
#             grid_size = (math.ceil((src.shape[1] - start_c) / block_size[0]),
#                                 math.ceil((src.shape[0] - start_r) / block_size[1]))
#             blk_count_nonzero[grid_size, block_size](d_src, zip_blk_x, zip_blk_y,
#                                                                             nonzero_count, (start_c, start_r))


#     threshold = mask_size * mask_size * zip_x * zip_y * ratio

#     for start_c in range(0, src.shape[1], zip_x):
#         for start_r in range(0, src.shape[0], zip_y):
#             grid_size = (math.ceil((src.shape[1] - start_c) / block_size[0]),
#                                 math.ceil((src.shape[0] - start_r) / block_size[1]))
#             apply_threshold[grid_size, block_size](d_src, threshold, nonzero_count, zip_blk_x, zip_blk_y,
#                                                                         (start_c, start_r), False)

#     return d_src

@cuda.jit
def convert_rgb2gray_kernel(in_pixels, out_pixels):
    c, r = cuda.grid(2)
    if r < out_pixels.shape[0] and c < out_pixels.shape[1]:
        out_pixels[r, c] = round(in_pixels[r, c, 0] * 0.114 + 
                            in_pixels[r, c, 1] * 0.587 + 
                            in_pixels[r, c, 2] * 0.299)

BLOCK_SIZE = (32, 32)
def convert_rgb2gray_use_kernel(image):
    # d_in_img = cuda.to_device(image)
    # d_gray_img = cuda.device_array((height, width), dtype=np.uint8)
    d_gray_img = cuda.device_array(image.shape[:2])

    grid_size = (math.ceil(image.shape[0] / BLOCK_SIZE[0]),
                        math.ceil(image.shape[1] / BLOCK_SIZE[1]))
    convert_rgb2gray_kernel[grid_size, BLOCK_SIZE](image, d_gray_img)
    
    # gray=d_gray_img.copy_to_host()
    return d_gray_img


def getFigureForImage3(path):
    img = cv.imread(path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = gray.astype(np.uint8)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # d_gray = gaussian_blur_kernel(gray, (3, 3), 0)
    # gray = d_gray.copy_to_host()

    mask_img = applyCannyThreshold(gray, 12)

    mask_img = zipImage(mask_img, 8, 8, 0.12)
    mask_img = zipImage(mask_img, 16, 16, 0.2)
    
    mask_img = joinNeighborPixel(mask_img, 8, 8, 3, 0.15)
    mask_img = joinNeighborPixel(mask_img, 16, 16, 3, 1 / 3)

    # for chanel in range(3):
    #     img[:,:,chanel] = img[:,:,chanel] * mask_img
    img *= np.atleast_3d(mask_img)

    hist_figure = fd_histogram2(img)
    hu_monents = fd_hu_moments2(gray)

    fig = np.concatenate((hist_figure, hu_monents))
    return fig

# test song song hóa
def compare_gray(path):
    img = cv.imread(path)
    height, width = img.shape[:2]

    gray_img = np.empty((height, width), dtype=img.dtype)

    gray1= convert_rgb2gray(img,gray_img)
    gray2= convert_rgb2gray_use_kernel(img)

    print('Convert rgb to grayscale error:',np.mean(np.abs(gray2- gray1)), '\n')
