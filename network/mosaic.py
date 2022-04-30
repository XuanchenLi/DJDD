import numpy as np
import torch


def bayer(img, return_mask=False):
    """
    G B
    R G
    """
    mask = torch.ones_like(img)

    # red
    mask[..., 0, ::2, :] = 0
    mask[..., 0, 1::2, 1::2] = 0

    # green
    mask[..., 1, 0::2, 1::2] = 0
    mask[..., 1, 1::2, 0::2] = 0

    # blue
    mask[..., 2, 1::2, :] = 0
    mask[..., 2, 0::2, 0::2] = 0

    if return_mask:
        return mask
    return img * mask


def bayer2(im, return_mask=False):
  """Bayer mosaic.
  The patterned assumed is::
    G r
    b G
  Args:
    im (np.array): image to mosaic. Dimensions are [c, h, w]
    return_mask (bool): if true return the binary mosaic mask, instead of the mosaic image.
  Returns:
    np.array: mosaicked image (if return_mask==False), or binary mask if (return_mask==True)
  """

  numpy = False
  if type(im) == np.ndarray:
    numpy = True

  if type(im) == np.ndarray:
    mask = np.ones_like(im)
  else:
    mask = torch.ones_like(im)

  # red
  mask[..., 0, ::2, 0::2] = 0
  mask[..., 0, 1::2, :] = 0

  # green
  mask[..., 1, ::2, 1::2] = 0
  mask[..., 1, 1::2, ::2] = 0

  # blue
  mask[..., 2, 0::2, :] = 0
  mask[..., 2, 1::2, 1::2] = 0

  if not numpy:  # make it a constant for ONNX conversion
    mask = torch.from_numpy(mask.cpu().detach().numpy()).to(im.device)

  if mask.shape[0] == 1:
    mask = mask.squeeze(0) # coreml hack

  if return_mask:
    return mask

  return im*mask


def bayer_down_sample(src):
    new_size = list(src.shape)
    new_size[1] = 4
    new_size[2] = int(new_size[2]/2)
    new_size[3] = int(new_size[3]/2)
    res = torch.zeros(tuple(new_size))
    res[:, 0, :, :] = src[:, 1, ::2, ::2]  # G
    res[:, 1, :, :] = src[:, 0, 1::2, ::2]  # R
    res[:, 2, :, :] = src[:, 2, ::2, 1::2]  # B
    res[:, 3, :, :] = src[:, 1, 1::2, 1::2]  # G
    return res


def bayer_up_sample(src):
    new_size = list(src.shape)
    new_size[1] = 3
    new_size[2] *= 2
    new_size[3] *= 2
    # print(src)
    res = torch.zeros(tuple(new_size))
    for c in range(3):
        res[:, c, ::2, ::2] = src[:, 4*c, :, :]
        res[:, c, ::2, 1::2] = src[:, 4*c+1, :, :]
        res[:, c, 1::2, ::2] = src[:, 4*c+2, :, :]
        res[:, c, 1::2, 1::2] = src[:, 4*c+3, :, :]
    return res


def bayer_down_sample2(src):
    new_size = list(src.shape)
    new_size[1] = 4
    new_size[2] = int(new_size[2]/2)
    new_size[3] = int(new_size[3]/2)
    res = torch.zeros(tuple(new_size))
    res[:, 0, :, :] = src[:, 1, ::2, ::2]  # G
    res[:, 1, :, :] = src[:, 0, ::2, 1::2]  # R
    res[:, 2, :, :] = src[:, 2, 1::2, ::2]  # B
    res[:, 3, :, :] = src[:, 1, 1::2, 1::2]  # G
    return res


def bayer_up_sample2(src):
    new_size = list(src.shape)
    new_size[1] = 3
    new_size[2] *= 2
    new_size[3] *= 2
    # print(src)
    res = torch.zeros(tuple(new_size))
    for c in range(3):
        res[:, c, ::2, ::2] = src[:, 4 * c, :, :]
        res[:, c, ::2, 1::2] = src[:, 4 * c + 1, :, :]
        res[:, c, 1::2, ::2] = src[:, 4 * c + 2, :, :]
        res[:, c, 1::2, 1::2] = src[:, 4 * c + 3, :, :]
    return res