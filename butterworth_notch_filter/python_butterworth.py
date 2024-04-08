### Manually implement Butterworth Notch Filter
### PTSA calls scipy.signal.butter and scipy.signal.filtfilt
### Let's manually write these in python
### and then translate into C++

# imports
import numpy as np
from scipy import linalg
import ctypes

def define_args(freq_range, sample_rate):
    """
    Calculate Nyquist frequency and scale frequency range 
    to pass proper arguments to Butterworth filter.

    Parameters
    ----------
    freq_range : array_like
        Length-2 sequence of critical frequencies = [58., 62.].
    sample_rate : int
        Sampling frequency of time-series = 1000 Hz.

    Returns
    -------
    wn : array_like
        Length-2 sequence of critical frequencies divided by 
        Nyquist frequency, to be passed as Wn argument of 
        Butterworth filter.
    """
    # Nyquist frequency
    nyq = sample_rate / 2.

    freq_range = np.asarray(freq_range)
    wn = freq_range / nyq
    return wn

# ---------- scipy.signal.butter ----------

def butter(N, Wn, btype='bandstop', analog=False, output='ba', fs=None):
    """
    Butterworth digital and analog filter design.

    Design Nth-order digital Butterworth filter and return 
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter = 4.
    Wn : array_like
        Length-2 sequence of critical frequencies divided by Nyquist 
        frequency = [58., 62.] / (sample_rate / 2).
    btype = 'bandstop'
        Type of filter = Butterworth notch filter.
    analog = False
        Digital filter.
    output = 'ba'
        Output in the form: numerator, denominator.
    fs = None
        Sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the 
        IIR filter.
    """
    return iirfilter(N, Wn, btype=btype, analog=analog, 
                     output=output, ftype='butter', fs=fs)

def iirfilter(N, Wn, rp=None, rs=None, btype='bandstop', analog=False, ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter = 4.
    Wn : array_like
        Length-2 sequence of critical frequencies divided by Nyquist 
        frequency = [58., 62.] / (sample_rate / 2).
    rp = None
        Not applicable for Butterworth filter.
    rs = None
        Not applicable for Butterworth filter.
    btype = 'bandstop'.
        Type of filter = Butterworth notch filter.
    analog = False
        Digital filter.
    ftype = 'butter'
        Type of IIR filter to design.
    output = 'ba'
        Filter form of the output: numerator, denominator.
    fs : float
        Sampling frequency of digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
    """
    ftype, btype, output = (x.lower() for x in (ftype, btype, output))
    Wn = np.asarray(Wn)
    
    btype = band_dict[btype]
    typefunc = filter_dict[ftype][0]

    # get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    else:
        raise ValueError("'%s' should be 'butter'." % ftype)
    
    # pre-warp frequencies for digital filter design
    if not analog:
        if np.any(Wn <= 0) or np.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies must "
                                 f"be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs/2})")
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * np.tan(np.pi * Wn / fs)
    else:
        warped = Wn

    # transform to bandstop
    if btype in ('bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = np.sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError('Wn must specify start and stop frequencies for '
                             'bandstop filter') from e
        
        if btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' should be 'bandstop'." % btype)
    
    # find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # transform to proper out type (numer-denom)
    if output == 'ba':
        return zpk2tf(z, p, k)
    
def buttap(N):
    """
    Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.e., rad/s) cutoff frequency of 1.
    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer.")
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # middle value is 0 to ensure exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1
    return z, p, k

def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    """
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and 
    stopnband width `bw` from an analog low-pass filter prototype with unity 
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = np.concatenate((z_hp + np.sqrt(z_hp**2 - wo**2),
                            z_hp - np.sqrt(z_hp**2 - wo**2)))
    p_bs = np.concatenate((p_hp + np.sqrt(p_hp**2 - wo**2),
                            p_hp - np.sqrt(p_hp**2 - wo**2)))
    
    # move any zeroes that were at infinity to the center of the stopband
    z_bs = np.append(z_bs, np.full(degree, +1j*wo))
    z_bs = np.append(z_bs, np.full(degree, -1j*wo))

    # cancel out gain change cause by inversion
    k_bs = k * np.real(np.prod(-z) / np.prod(-p))

    return z_bs, p_bs, k_bs

def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles.
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                            "Must have at least as many poles as zeros.")
    else:
        return degree
    
def bilinear_zpk(z, p, k, fs):
    """
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital 
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for 
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g, hz).
        No prewraping is done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # bilinear transfrom the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # any zeroes that were at infinity get moved to the Nyquist frequency
    z_z = np.append(z_z, -np.ones(degree))

    # compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_z, p_z, k_z

def zpk2tf(z, p, k):
    """
    Return polynomial transfer function representation from zeros and poles.

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominatory polynomial coefficients.
    """
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    if len(z.shape) > 1:
        temp = np.poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0] * z.shape[0]]
        for i in range(z.shape[0]):
            b[i] = k[i] * np.poly(z[i])
    else:
        b = k * np.poly(z)
    a = np.atleast_1d(np.poly(p))

    # use real output if possible
    # copied from np.poly, since we can't depend on a specific version of numpy
    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == 
                        np.sort_complex(pos_roots)):
                b = b.real.copy()
    
    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == 
                        np.sort_complex(pos_roots)):
                a = a.real.copy()
    
    return b, a

# ---------- scipy.signal.filtfilt ----------

def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):
    """
    Apply a digital filter forward and backward to a signal.

    This function applies a linear digital filter twice, once forward and 
    once backwards.  The combined filter has zero phase and a filter order 
    twice that of the original.

    The function provides options for handling the edges of the signal.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]`` 
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        The array of data to be filtered.
    axis = -1
        The axis of `x` to which the filter is applied.
    padtype = 'odd'
        Determines the type of extension to use for the padded signal 
        to which the filter is applied.
    padlen = None
        The number of elements by which to extend `x` at both ends of 
        `axis` before applying the filter.
        The default value is ``3 * max(len(a), len(b))``.
    method = 'pad'
        Determines the method for handling the edges of the signal.
        When `method` is "pad", the signal is padded; the type of padding 
        os determined by `padtype` and `padlen`, and `irlen` is ignored.
    irlen = None
        If `irlen` is None, no part of the impulse response is ignored.
        For a long signal, specifying `irlen` can significantly imporve 
        the performance of the filter.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
    """
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    x = np.asarray(x)

    if method not in ["pad"]:
        raise ValueError("method must be 'pad'.")
    
    edge, ext = _validate_pad(padtype, padlen, x, axis, 
                                ntaps=max(len(a), len(b)))
    
    # get the steady state of the filter's step response
    zi = lfilter_zi(b, a)

    # reshape zi and create x0 so that zi*xo broadcasts to the
    # correct value for the `zi` keyword argument to lfilter
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # forward filter
    (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # backward filter
    # create y0 so zi*yo broadcasts appropriately
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # reverse y
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # slice the actual signal from the extended signal
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y

def _validate_pad(padtype, padlen, x, axis, ntaps):
    """
    Helper to validate padding for filtfilt.
    """
    if padtype not in ['odd']:
        raise ValueError(("Unknown value '%s' given to padtype. "
                            "padtype must be 'odd'.") % padtype)
    
    edge = padlen
    
    # x's 'axis' dimension must be bigger than edge
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be greater "
                            "than padlen, which is %d." % edge)
    
    if padtype is not None and edge > 0:
        # make an extension of length `edge` at each end of the input array
        if padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            raise ValueError("padtype should be 'odd")
    else:
        ext = x
    
    return edge, ext

def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    """
    Take a slice along axis 'axis' from 'a'.

    Parameters
    ----------
    a : np.ndarray
        The array to be sliced.
    start, stop, step: int or None
        The slice parameters.
    axis = -1
        The axis to `a` to be sliced.
    """
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]

    return b

def axis_reverse(a, axis=-1):
    """
    Reverse the 1-D slices of `a` along axis `axis`.

    Returns axis_slice(a, step=-1, axis=axis).
    """
    return axis_slice(a, step=-1, axis=axis)

def odd_ext(x, n, axis=-1):
    """
    Odd extension at the boundaries of an array.

    Generate a new ndarray by making an odd extension of `x` along an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis = -1
        The axis along which to extend `x`.
    """
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extend length n (%d) is too big. " + 
                            "It must not exceed x.shape[axis]-1, which is %d.")
                            % (n, x.shape[axis] - 1))
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n+2), step=-1, axis=axis)
    ext = np.concatenate((2 * left_end - left_ext, x, 
                            2 * right_end - right_ext), axis=axis)
    
    return ext

def lfilter_zi(b, a):
    """
    Construct inital conditions for lfilter for step response steady-state.

    Compute an initial state `zi` for the `lfilter` function that corresponse 
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the 
    output of the filter starts at the same value as the first element of 
    the signal to be filtered.

    Parameters
    ----------
    b, a : array_like (1-D)
        The IIR filter coefficients.  See `lfilter` for more information.

    Returns
    -------
    zi : 1-D ndarray
        The inital state for the filter.
    """
    # We could use scipy.siginal.normalize, but it uses warnings in cases
    # where a ValueError is more appropriate, and it allows b to be 2D
    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("Numerator b must be 1-D.")
    a = np.atleast_1d(a)
    if a.ndim != 1:
        raise ValueError("Denominator a must be 1-D.")
    
    while len(a) > 1 and a[0] == 0.0:
        a = a[1:]
    if a.size < 1:
        raise ValueError("There must be at least one nonzero `a` coefficient.")
    
    if a[0] != 1.0:
        # normalize the coefficients so a[0] == 1
        b = b / a[0]
        a = a / a[0]

    n = max(len(a), len(b))

    # pad a or b with zeros so they are the same length
    if len(a) < n:
        a = np.r_[a, np.zeros(n - len(a), dtype=a.dtype)]
    elif len(b) < n:
        b = np.r_[b, np.zeros(n - len(b), dtype=b.dtype)]

    IminusA = np.eye(n - 1, dtype=np.result_type(a, b)) - linalg.companion(a).T
    B = b[1:] - a[1:] * b[0]
    
    # solve zi = A*zi + B
    zi = np.linalg.solve(IminusA, B)

    return zi

def lfilter(b, a, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence, `x`, using a digital filter.  This words for many 
    fundamental data types (including Object type).  The filter is a direct 
    form II transposed implementation of the standard difference equation.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]`` 
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        An N-dimensional input array.
    axis = -1
        The axis of the input data array along which to apply the 
        linear filter.  The filter is applied to each subarray along 
        this axis.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector 
        (or array of vectors for an N-dimensional input) of length 
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then 
        initial rest is assumed.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the 
        final filter delay values.
    """
    a = np.atleast_1d(a)
    if len(a) == 1:
        b = np.asarray(b)
        a = np.asarray(a)
        if b.ndim != 1 and a.ndim != 1:
            raise ValueError('object of too small depth for desired array')
        x = _validate_x(x)
        inputs = [b, a, x]
        if zi is not None:
            zi = np.asarray(zi)
            if zi.ndim != x.ndim:
                raise ValueError('object of too small depth for desired array')
            expected_shape = list(x.shape)
            expected_shape[axis] = b.shape[0] - 1
            expected_shape = tuple(expected_shape)

            # check trivial case where zi is the right shape first
            if zi.shape != expected_shape:
                strides = zi.ndim * [None]
                if axis < 0:
                    axis += zi.ndim
                for k in range(zi.ndim):
                    if k == axis and zi.shape[k] == expected_shape[k]:
                        strides[k] = zi.strides[k]
                    elif k != axis and zi.shape[k] == expected_shape[k]:
                        strides[k] = zi.strides[k]
                    elif k != axis and zi.shape[k] == 1:
                        strides[k] = 0
                    else:
                        raise ValueError('Unexpected shape for zi: expected '
                                            f'{expected_shape}, found {zi.shape}.')
                zi = np.lib.stride_tricks.as_strided(zi, expected_shape, strides)
            
            inputs.append(zi)
        dtype = np.result_type(*inputs)

        if dtype.char not in 'fdgFDGO':
            raise NotImplementedError("input type '%s' not supported" % dtype)
        
        b = np.array(b, dtype=dtype)
        a = np.array(a, dtype=dtype, copy=False)
        b /= a[0]
        x = np.array(x, dtype=dtype, copy=False)

        out_full = np.apply_along_axis(lambda y: np.convolve(b, y), axis, x)
        ind = out_full.ndim * [slice(None)]
        if zi is not None:
            ind[axis] = slice(zi.shape[axis])
            out_full[tuple(ind)] += zi

        ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
        out = out_full[tuple(ind)]

        if zi is None:
            return out
        else:
            ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
            zf = out_full[tuple(ind)]
            return out, zf
    else:
        if zi is None:
            # the _linear_filter method is in c++
            return _linear_filter(b, a, x, axis)
        else:
            return _linear_filter(b, a, x, axis, zi)
        
def _validate_x(x):
    x = np.asarry(x)
    if x.ndim == 0:
        raise ValueError('x must be at least 1-D')
    return x
            
def _linear_filter(b, a, x, axis, zi):
    scipy_signal__sigtools_linear_filter_module_init()
    scipy_signal__sigtools_linear_filter()
    #raise NotImplementedError("Scipy implements in c++")

filter_dict = {'butter': [buttap],
               'butterworth': [buttap],
               }

band_dict = {'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',
             }

# ---------- scipy.signal linear_filter (C) ----------

def scipy_signal__sigtools_linear_filter_module_init():
    BasicFilterFunctions = {}
    for k in range(256):
        BasicFilterFunctions[k] = None
    
    # only need float and double
    BasicFilterFunctions[np.float] = FLOAT_filt
    BasicFilterFunctions[np.double] = DOUBLE_filt

def scipy_signal__sigtools_linear_filter():
    def fail(ara, arb, arX, arVi, arVf, arY):
        del ara
        del arb
        del arX
        del arVi
        del arVf
        del arY
        return None
    # b, a, X, Vi
    # arY, arb, ara, arX, arVi, arVf
    # axis, typenum, theaxis, st, Vi_needs_broadcasted = 0
    # ara_ptr, input_flag = 0, azero
    # na, nb, nal, zi_size
    # zf_shape
    # basic_filter
    
    axis = -1
    Vi = None
    
    # translated by chatgpt
    typenum = np.array(b, dtype=int).dtype
    typenum = np.array(a, dtype=typenum).dtype
    typenum = np.array(X, dtype=typenum).dtype
    if Vi != None:
        typenum = np.array(Vi, typenum).dtype
        
    arY = arVf = arVi = None
    
    # translated by chatgpt
    ara = np.ascontiguousarray(a, dtype=typenum)
    arb = np.ascontguousarray(b, dtype=typenum)
    arX = np.array(X, dtype=typenum, copy=False, order='K')
    
    if ara == None or arb == None or arX == None:
        fail(ara, arb, arX, arVi, arVf, arY)
        raise ValueError("Could not convert b, a, and x to a common type")
        
    if axis < -arX.ndim or axis > axX.ndim - 1:
        fail(ara, arb, arX, arVi, arVf, arY)
        raise ValueError("Selected axis is out of range")
        
    if axis < 0:
        theaxis = arX.ndim + axis
    else:
        theaxis = axis
        
    if Vi != None:
        # translated by chatgpt
        ndim_arX = np.ndim(arX)
        arVi = np.array(Vi, dtype=typenum, copy=False, ndmin=ndim_arX)
        
        if arVi == None:
            fail(ara, arb, arX, arVi, arVf, arY)
            
        input_flag = 1
        
    # translated by chatgpt
    arX_dtype = arX.dtype
    if arX_dtype < np.dtype('uint8'):
        basic_filter = BasicFilterFunctions[arX.dtype.type]
    else:
        basic_filter = None
        
    if basic_filter == None:
        fail(ara, arb, arX, arVi, arVf, arY)
        
    # skip over leading zeros in vector representing denominator (a)
    azero = np.zeros_like(ara)
    if azero == None:
        fail(ara, arb, arX, arVi, arVf, arY)
        
    # translated by chatgpt
    ara_ptr = ara.ctypes.data
    nal = ara.itemsize
    st = ctypes.memcmp(ara_ptr, azero.data, nal)
    azero = None
    if st == 0:
        fail(ara, arb, arX, arVi, arVf, arY)
        raise ValueError("filter coefficient a[0] == 0 not supported")
        
    na = ara.size
    nb = arb.size
    # translated by chatgpt
    zi_size = max(na, nb) - 1
    
    if input_flag:
        # npy_intp k, Vik, Xk
        for k in range(arX.ndim):
            # translated by chatgpt
            Vik = arVi.shape[k]
            Xk = arX.shape[k]
            if k == theaxis and Vik == zi_size:
                zf_shape[k] = zi_size
            elif k != theaxis and Vik == Xk:
                zf_shape[k] = Xk
            elif k != theaxis and Vik == 1:
                zf_shape[k] = Xk
                Vi_needs_broadcasted = 1
            else:
                fail(ara, arb, arX, arVi, arVf, arY)
                
        if Vi_needs_broadcasted:
            # arVi_view
            # view_dtype
            # translated by chatgpt
            arVi_shape = arVI.shape
            arVi_strides = arVi.strides
            ndim = arVi.ndim
            strides = np.empty(ndim)
            # strides
            # k
            
            for k in range(ndim):
                if arVi.shape[k] == 1:
                    strides[k] = 0
                else:
                    strides[k] = arVi_strides[k]
                    
            # translated by chatgpt
            view_dtype = arVi.dtype
            
            # translated by chatgpt
            arVi_view = np.ndarray((0,), dtype=view_dtype, buffer=arVi, strides=strides, offset=0)
            if not arVi_view:
                fail(ara, arb, arX, arVi, arVf, arY)
                
            if np.base_repr(arVi_view, arVi) == -1:
                arVi_view = None
                fail(ara, arb, arX, arVi, arVf, arY)
            
            arVi = arVi_view
        
        # translated by chatgpt
        arVf = np.zeros(zf_shape, dtype=typenum)
        
        if not arVF:
            fail(ara, arb, arX, arVi, arVf, arY)
            
        # translated by chatgpt
        arY = np.zeros(arX.shape, dtype=typenum)
        
        if not arY:
            fail(ara, arb, arX, arVi, arVf, arY)
            
        st = RawFilter(arb, ara, arX, arVi, arVf, arY, theaxis, basic_filter)
        if st:
            fail(ara, arb, arX, arVi, arVf, arY)
            
        # translated by chatgpt
        del ara
        del arb
        del arX
        del arVi
        
        if not input_flag:
            return arY
        else:
            # translated by chatgpt
            result_tuple = (arY, arVf)
            return result_tuple

def RawFilter(b, a, x, zi, zf, y, axis, filter_func):
    def clean_itx():
        del itx
        
    def clean_ity():
        del ity
        
    def clean_itzi(zi, itzi):
        if zi:
            del itzi
            
    def clean_itzf(zf, itzf):
        if zi:
            del itzi
            
    def clean_azfilled():
        azfilled = None
        
    def clean_bzfilled():
        bzfilled = None
        
    def clean_zfzfilled():
        zfzfilled = None
        
    def fail():
        return -1
    
    # itx, itx
    itzi = None
    itzf = None
    # nitx, i, nxl, nzfl, j
    # na, nb, nal, nbl
    # nfilt
    # azfilled, bzfilled, zfzfilled, yoyo
    # translated by chatgpt
    copyswap = x.dtype.descr.f.copyswap
    itx = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'], op_axes=[axis])
    
    if not itx:
        fail()
        raise RuntimeError("Could not create itx")
        
    # translated by chatgpt
    nitx = itx.size
    
    # translated by chatgpt
    ity = np.nditer(y, flags=['multi_index'], op_flags=['readwrite'], op_axes=[axis])
    
    if not ity:
        clean_itx()
        raise RuntimeError("Could not create ity")
    
    if zi:
        # translated by chatgpt
        itzi = np.nditer(zi, flags=['multi_index'], op_flags=['readwrite'], op_axes=[axis])
        if not itzi:
            clean_ity()
            raise RuntimeError("Could not create itzi")
            
        # translated by chatgpt
        itzf = np.nditer(zf, flags=['multi_index'], op_flags=['readwrite'], op_axes=[axis])
        
        if not itzf:
            clean_itzi(zi, itzi)
            raise RuntimeError("Could not create itzf")
       
    # translated by chatgpt
    na = np.size(a)
    nal = a.itemsize
    nb = np.size(b)
    nbl = b.itemsize
    
    # translated by chatgpt
    nfilt = max(na, nb)
    
    # translated by chatgpt
    azfilled = np.empty(nal * nfilt)
    if not azfilled:
        clean_itzf(zf, itzf)
        raise RuntimeError("Could not create azfilled")
        
    bzfilled = np.empty(nbl * nfilt)
    if not bzfilled:
        clean_azfilled()
        raise RuntimeError("Could not create bzfilled")
        
    nxl = x.itemsize
    zfzfilled = np.empty(nxl * (nfilt - 1))
    if not zfzfilled:
        clean_bzfilled()
        raise RuntimeError("Could not create zfzfilled")
        
    # translated by chatgpt
    azfilled[:] = 0
    bzfilled[:] = 0
    zfzfilled[:nxl * (nfilt - 1)] = 0
    
    if zfill(a, na, azfilled, nfilt) == -1:
        clean_zfzfilled()
    if zfill(b, nb, bzfilled, nfilt) == -1:
        clean_zfzfilled()
        
    if zf:
        nzfl = zf.itemsize
    else:
        nzfl = 0
        
    # iterate over input array
    for i in range(nitx):
        if zi:
            # translated by chatgpt
            yoyo = itzi.dataptr
            
            # copy initial conditions zi in zfzfilled buffer
            for j in range(nfilt - 1):
                # translated by chatgpt
                np.copyto(zfzfilled[j * nzfl:], yoyo)
                yoyo += itzi.strides[axis]
                
            # translated by chatgpt
            next(itzi)
        else:
            if zfill(x, 0, zfzfilled, nfilt - 1) == -1:
                clean_zfzfilled()
                
        try:
            filter_func(bzfilled, azfilled, its.dataptr, ity.dataptr, nfilt, x.shape[axis], 
                        itz.strides[axis], ity.strides[axis])
        except BaseException:
            clean_zfzfilled()
            
        next(itx)
        next(ity)
        
        # copy tmp buffer of final values back into zf output array
        if zi:
            yoyo = itzf.dataptr
            for j in range(nfilt - 1):
                # translated by chatgpt
                np.copyto(yoyo, zfzfilled[l * nzfl:])
                yoy += itzf.strides[axis]
                
            next(itzf)
            
    # free up allocated memory
    zfzfilled = None
    bzfilled = None
    azfilled = None
    
    if zi:
        del itzf
        del itzi
    
    del itz
    del itx
    
    return 0
    
    
def zfill(x, nx, xzfilled, nxzfilled):
    # xzero
    # i, nxl
    # translated by chatgpt
    copyswap = x.dtype.descr.f.copyswap
    
    nxl = x.itemsize
    
    xzero = np.zeros_like(x)
    if not xzero:
        return -1
    
    if nx > 0:
        for i in range(nx):
            # translated by chatgpt
            np.copyto(xzfilled[i * nxl:], x.data[i * nxl:])
    
    # translated by chatgpt
    for i in range(nx, nxzfilled):
        np.copyto(xzfilled[i * nxl:], xzero)
        
    xzero = None
    
    return 0
    
# could be either DOUBLE_filt or FLOAT_filt
def filter_func(b, a, x, y, z, len_b, len_x, stride_X, stride_Y):
    ptr_x = x
    ptr_y = y
    # ptr_Z
    ptr_b = b
    ptr_a = a
    # xn, yn
    a0 = a
    # n
    # k
    
    # normalize the filter coefficients only once
    for n in range(len_b):
        ptr_b[n] /= a0
        ptr_a[n] /= a0
        
    for k in range(len_x):
        ptr_b = b
        ptr_a = a
        xn = ptr_x
        yn = ptr_y
        if len_b > 1:
            ptr_Z = Z
            yn = ptr_z + ptr_b * xn   # calculate first delay (output)
            ptr_b += 1
            ptr_a += 1
            
            # fill in middle delays
            for n in range(len_b - 2):
                ptr_Z = ptr_Z[1] + xn * ptr_b - yn * ptr_a
                ptr_b += 1
                ptr_a += 1
                ptr_Z += 1
                
            # calculate last delay
            ptr_Z = xn * ptr_b - yn * ptr_a
        else:
            yn = xn * ptr_b
            
        # move to next input/output point
        ptr_y += stride_Y
        ptr_x += stride_X
