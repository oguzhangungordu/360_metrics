import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math
import glob as gb
import os.path as osp
from skimage.metrics import structural_similarity as ssim
import struct

def cal_ws_metrics(src_gt, src_sr,src_sal, metrics=None, print_avg=True, print_details=True):
    '''
    :param src_gt (str): Source path of gt images
    :param src_sr (str): Source path of sr images
    :param metrics (list): Select metrics within wspsnr and wsssim
    :param print_avg (bool): Whether to print average values after calculation
    :param print_details (bool): Whether to print detail values during calculation
    :return:
        output dict with detailed values and average values
    '''
    if metrics is None:
        metrics = ['wspsnr', 'wsssim']

    pths_gt = sorted(gb.glob(gb.escape(src_gt)+'/*'))
    pths_sr = sorted(gb.glob(gb.escape(src_sr)+'/*'))
    pths_sal = sorted(gb.glob(gb.escape(src_sal)+'/*'))

    output_dict = {}
    for metric in metrics:
        output_dict[metric] = {}
        output_dict[metric+'_avg'] = 0

    assert len(pths_gt) == len(pths_sr)
    assert len(pths_gt) == len(pths_sal)
    n = len(pths_gt)

    for i, (pth_gt, pth_sr, pth_sal) in enumerate(zip(pths_gt, pths_sr,pths_sal)):
        idx = osp.splitext(osp.split(pth_gt)[1])[0]
        assert idx in osp.splitext(osp.split(pth_sr)[1])[0]
        img_gt = cv2.imread(pth_gt)
        img_sr = cv2.imread(pth_sr)
        #img_sal = cv2.imread(pth_sal)

        file_size_compressed = osp.getsize(pth_sr)
        height_compressed, width_compressed, channels = img_sr.shape

        with open(pth_sal, 'rb') as fileId:
          buf = fileId.read(width_compressed * height_compressed * 4)  # 4 bytes for single precision float
          img_sal = struct.unpack('f' * (width_compressed * height_compressed), buf)
          img_sal = np.array(img_sal).reshape((height_compressed, width_compressed))

        file_size_compressed = osp.getsize(pth_sr)
        height_compressed, width_compressed, channels = img_sr.shape

        s_psnr = ''
        if 'wspsnr' in metrics:
            _output = calculate_psnr_ws(img_gt, img_sr, crop_border=0)
            output_dict['wspsnr'][idx] = _output
            output_dict['wspsnr_avg'] += _output/n
            s_psnr = f'\twspsnr:{_output:.2f}'

        s_ssim = ''
        if 'wsssim' in metrics:
            _output = calculate_ssim_ws(img_gt, img_sr, crop_border=0)
            output_dict['wsssim'][idx] = _output
            output_dict['wsssim_avg'] += _output/n
            s_ssim = f'\twsssim:{_output:.4f}'
        
        sal_psnr = ''
        if 'salpsnr' in metrics:
            _output = calculate_sal_psnr_ws(img_gt, img_sr, img_sal, crop_border=0)
            output_dict['salpsnr'][idx] = _output
            output_dict['salpsnr_avg'] += _output/n
            sal_psnr = f'\tsalpsnr:{_output:.4f}'

        psnr = ''
        if 'psnr' in metrics:
            _output = calculate_psnr(img_gt, img_sr, crop_border=0)
            output_dict['psnr'][idx] = _output
            output_dict['psnr_avg'] += _output/n
            psnr = f'\tpsnr:{_output:.4f}'

        ssim = ''
        if 'ssim' in metrics:
            _output = calculate_ssim(img_gt, img_sr, crop_border=0)
            output_dict['ssim'][idx] = _output
            output_dict['ssim_avg'] += _output/n
            ssim = f'\tssim:{_output:.4f}'

        bpp = ''
        if 'bpp' in metrics:
            _output = (file_size_compressed * 8) / (3*width_compressed * height_compressed)
            output_dict['bpp'][idx] = _output
            output_dict['bpp_avg'] += _output/n
            bpp = f'\tbpp:{_output:.4f}'

        if print_details:
            print(f'[{i}/{n}][idx]:{s_psnr}{s_ssim}{sal_psnr}{psnr}{ssim}{bpp}')

    if print_avg:
        if 'wspsnr' in metrics:
            print(f'Average value of WS-PSNR:\t {output_dict["wspsnr_avg"]}')
        if 'wsssim' in metrics:
            print(f'Average value of WS-SSIM:\t {output_dict["wsssim_avg"]}')
        if 'salpsnr' in metrics:
            print(f'Average value of SAL-PSNR:\t {output_dict["salpsnr_avg"]}')
        if 'psnr' in metrics:
            print(f'Average value of PSNR:\t {output_dict["psnr_avg"]}')  
        if 'ssim' in metrics:
            print(f'Average value of SSIM:\t {output_dict["ssim_avg"]}') 
        if 'bpp' in metrics:
            print(f'Average value of bpp:\t {output_dict["bpp_avg"]}')  

    return output_dict


def calculate_psnr_ws(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate WS-PSNR.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: WS-PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    img_w = compute_map_ws(img)

    mse = np.mean(np.multiply((img - img2)**2, img_w))/np.mean(img_w)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)

def calculate_psnr(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate WS-PSNR.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: WS-PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)

def calculate_sal_psnr_ws(img, img2, sal_map, crop_border, input_order='HWC', **kwargs):
    """Calculate WS-PSNR.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: WS-PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    sal_map_3channel = np.stack([sal_map, sal_map, sal_map], axis=-1)
    img_w = np.multiply(compute_map_ws(img),sal_map_3channel)

    mse = np.mean(np.multiply((img - img2)**2, img_w))/np.mean(img_w)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)



def calculate_ssim_ws(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ws_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()

def calculate_ssim(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssim_ = ssim(img, img2, multichannel=True)
    return ssim_


def genERP(j,N):
    val = math.pi/N
    w = math.cos((j - (N/2) + 0.5) * val)
    return w

def compute_map_ws(img):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img(HWC)    the input original video
    """
    equ = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for i in range(0,equ.shape[0]):
        for j in range(0,equ.shape[1]):
            for k in range(0,equ.shape[2]):
                equ[i, j, k] = genERP(i,equ.shape[0])
    return equ

def _ws_ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    equ = np.zeros((ssim_map.shape[0], ssim_map.shape[1]))

    for i in range(0,equ.shape[0]):
        for j in range(0,equ.shape[1]):
                equ[i, j] = genERP(i,equ.shape[0])

    return np.multiply(ssim_map, equ).mean()/equ.mean()


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img
