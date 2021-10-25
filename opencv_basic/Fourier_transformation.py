# 傅里叶变换的作用:
# 图像傅里叶变换后低频在四周、高频在中心(经过fftshift后，低频在中间，高频在四周)
# 高频：变化剧烈的灰度分量，例如边界
# 低频：变化缓慢的灰度分量，例如一片大海
# 低通滤波器：只保留低频，会使得图像模糊
# 高通滤波器：只保留高频，会使得图像细节增强

import cv2 as cv
import numpy as np
from image_merge import plt_pic


# 快速傅里叶变换
def fast_fourier(img_path):
    image = cv.imread(img_path)

    if image is None:
        print("无法读取图片")
        return -1

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)    # 进行傅里叶变换时，需要先将图像转换成np.float32 格式
    dft = cv.dft(gray_float32, flags=cv.DFT_COMPLEX_OUTPUT)   # cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）
    dftShift = np.fft.fftshift(dft)                # 得到的结果中频率为0的部分会在左上角，通常要转换到中心位置
    magnitude_spectrum = 20 * np.log(cv.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))

    # 掩膜
    rows, cols = gray.shape
    rows_half, cols_half = int(rows / 2), int(cols / 2)  # 确定图像中心点位置
    mask_lowF = np.zeros((rows, cols, 2), dtype=np.uint8)     # 因为傅里叶变换为三维，所以掩膜也必须是三维的
    mask_lowF[rows_half - 30:rows_half + 30, cols_half - 30:cols_half + 30] = 1   # 低通滤波
    mask_highF = np.ones((rows, cols, 2), dtype=np.uint8)
    mask_highF[rows_half - 30:rows_half + 30, cols_half - 30:cols_half + 30] = 0   # 高通滤波

    # 逆傅里叶变换
    fShift_lowF = dftShift * mask_lowF   # 矩阵*代表对应元素相乘
    ishift_lowF = np.fft.ifftshift(fShift_lowF)
    iimg_lowF = cv.idft(ishift_lowF)
    dst_lowF = cv.magnitude(iimg_lowF[:, :, 0], iimg_lowF[:, :, 1])

    # 逆傅里叶变换
    fShift_highF = dftShift * mask_highF
    ishift_highF = np.fft.ifftshift(fShift_highF)
    iimg_highF = cv.idft(ishift_highF)
    dst_highF = cv.magnitude(iimg_highF[:, :, 0], iimg_highF[:, :, 1])

    image_dict = {"original": image, "快速傅里叶变换": magnitude_spectrum,  "低频滤波": dst_lowF, "高频滤波": dst_highF}
    plt_pic(image_dict)


if __name__ == "__main__":
    image_path = r"./images/a_zhu.jpg"
    fast_fourier(image_path)

