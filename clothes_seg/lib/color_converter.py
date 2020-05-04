#!/usr/bin/python
# package scikit-image/skimage/color/colorconv.py
import numpy as np 

def rgb2hsv(rgb):
    """RGB to HSV color space conversion.
    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape ``(.., .., 3)``.
    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    Examples
    --------
    >>> from skimage import color
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_hsv = color.rgb2hsv(img)
    """
    arr = _prepare_colorarray(rgb)
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.] = 0.

    # -- H channel
    # red is max
    idx = (arr[:, :, 0] == out_v)
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:, :, 1] == out_v)
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = (arr[:, :, 2] == out_v)
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    np.seterr(**old_settings)

    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0

    return out


def hsv2rgb(hsv):
    """HSV to RGB color space conversion.
    Parameters
    ----------
    hsv : array_like
        The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `hsv` is not a 3-D array of shape ``(.., .., 3)``.
    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_hsv = rgb2hsv(img)
    >>> img_rgb = hsv2rgb(img_hsv)
    """
    arr = _prepare_colorarray(hsv)

    hi = np.floor(arr[:, :, 0] * 6)
    f = arr[:, :, 0] * 6 - hi
    p = arr[:, :, 2] * (1 - arr[:, :, 1])
    q = arr[:, :, 2] * (1 - f * arr[:, :, 1])
    t = arr[:, :, 2] * (1 - (1 - f) * arr[:, :, 1])
    v = arr[:, :, 2]

    hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    out = np.choose(hi, [np.dstack((v, t, p)),
                         np.dstack((q, v, p)),
                         np.dstack((p, v, t)),
                         np.dstack((p, q, v)),
                         np.dstack((t, p, v)),
                         np.dstack((v, p, q))])

    return out

def _prepare_colorarray(arr):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
        msg = ("the input array must be have a shape == (.., ..,[ ..,] 3)), " +
               "got (" + (", ".join(map(str, arr.shape))) + ")")
        raise ValueError(msg)

    return dtype.img_as_float(arr)