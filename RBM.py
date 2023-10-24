from __future__ import annotations
from typing import Final, Literal
import numpy as np
from scipy.special import expit  # sigmoid function


# some useful reference
# https://github.com/isabekov/GBPRBM_NumPy/blob/master/GBPRBM.py
# https://medium.com/datatype/restricted-boltzmann-machine-a-complete-analysis-part-3-contrastive-divergence-algorithm-3d06bbebb10c
class RBM:
    def __init__(self, visible_layer_size: int, hidden_layer_size: int) -> None:
        # TODO verify if randomly initializing the weights in [0,1) makes sense
        self.weights_array: np.ndarray = np.random.rand(
            visible_layer_size, hidden_layer_size
        )  # model weights
        # note that visible units are initialized as ints
        self.visible_units_array: np.ndarray = np.zeros(
            (1, visible_layer_size), dtype=np.float2  # TODO int32 or float32?
        )  # N stochastic binary visible units
        self.hidden_units_array: np.ndarray = np.zeros(
            (1, hidden_layer_size), dtype=np.float32  # TODO int32 or float32?
        )  # M stochastic binary hidden units
        self.visible_layer_bias = np.zeros(
            (1, visible_layer_size), dtype=np.float32
        )  # bias for visible layer
        self.hidden_layer_bias = np.zeros(
            (1, hidden_layer_size), dtype=np.float32
        )  # bias for hidden layer

    def __repr__(self):
        """A convenient method to represent the model instance in terminal.
        Alternatively, is used with repr(self).

        Returns:
            str: a representation of the model class

        Sample Usage:
        You can directly get it in terminal:
        >>> RBM_instance = RBM(visible_layer_size=5, hidden_layer_size=6);
        >>> RBM_instance
        RBM(visible_layer_size=5, hidden_layer_size=6)
        """
        return f"RBM(visible_layer_size={self.weights_array.shape[0]}, hidden_layer_size={self.weights_array.shape[1]})"

    def __str__(self):
        """A convenient method to represent the model instance as a string.
        Alternatively, is used with str(self).
        Note that the float precision is set to 2 decimal places,
        and only the top 8 elements are printed for readability.

        Returns:
            str: a representation of the model class

        """
        # https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
        # with np.printoptions(precision=2, suppress=True, threshold=8):
        return f"""
        For {self!r}, the params are:
        {self.weights_array=}
        {self.visible_units_array=}
        {self.hidden_units_array=}
        {self.visible_layer_bias=}
        {self.hidden_layer_bias=}
        """

    def get_binary_state_of_hidden_units(self, sample_visible_units_array):
        # have (1,M)+(1,N)@(N,M), so shapes match
        return expit(
            self.hidden_layer_bias + sample_visible_units_array @ self.weights_array
        )

    def get_binary_state_of_visible_units(self, sample_hidden_units_array):
        # have (1,N)+(1,M)@(N,M)^T=(1,N)+(1,M)@(M,N), so shapes match
        return expit(
            self.visible_layer_bias + sample_hidden_units_array @ self.weights_array.T
        )

    def train_via_k_step_contrastive_divergence(
        self, k: int, S: np.ndarray, learning_rate: float
    ):
        # initialization
        delta_visible_units_array = np.zeros_like(
            self.visible_units_array
        )  # delta_a with length=N
        delta_hidden_units_array = np.zeros_like(
            self.hidden_units_array
        )  # delta_b with length=M
        delta_weights = np.zeros_like(self.weights_array)  # delta_W with a length=N*M

        # Let v = (v1, . . . , vN ) and h =(h1, . . . , hM)
        for current_sample in S:
            next_hidden_unit_values = np.zeros_like(self.hidden_units_array, dtype=bool)
            next_visible_unit_values = current_sample
            for i in range(k):
                # all units with prob>0.5 will be assigned a value of 1, 0 otherwise
                # we cast to int to convert from a boolean to an integer array
                next_hidden_unit_values = (
                    self.get_binary_state_of_hidden_units(next_visible_unit_values)
                    > 0.5
                ).astype(int)
                next_visible_unit_values = (
                    self.get_binary_state_of_visible_units(next_hidden_unit_values)
                    > 0.5
                ).astype(int)

            # update gradients
            delta_weights += learning_rate * (
                self.get_binary_state_of_hidden_units(current_sample) * current_sample
                - self.get_binary_state_of_hidden_units(next_visible_unit_values)
                * next_visible_unit_values
            )
            delta_visible_units_array += learning_rate * (
                current_sample - next_visible_unit_values
            )
            delta_hidden_units_array += learning_rate * (
                self.get_binary_state_of_hidden_units(current_sample)
                - self.get_binary_state_of_hidden_units(next_visible_unit_values)
            )

            # TODO realistically, need to verify how visible/hidden units updates behave if
            # their types are ints while learning_rate * delta_visible_units_array/
            # learning_rate * delta_hidden_units_array is a float number;
            # update weights
            self.weights_array += learning_rate * delta_weights
            self.visible_units_array += learning_rate * delta_visible_units_array
            self.hidden_units_array += learning_rate * delta_hidden_units_array


def convert_uint_array_to_binary(
    uint_array: np.ndarray,
    num_bits: int = 16,
    bit_order: Literal["big_first", "small_first"] = "small_first",
) -> np.ndarray:
    """Convert an array of (unsigned) integers to an array of bits()
    Refer to https://stackoverflow.com/a/51509307 for more info

    Args:
        uint_array (np.ndarray): an unsigned integer array.
            The values must not exceed 2^num_bits-1 to enable the conversion.
        num_bits (int): a positive biy number to convert to.
        bit_order (Literal['big_first','small_first'], optional): Bit order of numbers. Defaults to 'small_first'.

    Returns:
        np.ndarray: an array of ints corresponding to the number of bits that was requested.
            Low bit first by default, high bit first if requested

    Sample Usage:
    >>> print(convert_uint_array_to_binary(np.array([1, 2, 16, 2**16 - 1])))  # doctest: +NORMALIZE_WHITESPACE
        [[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]

    Note how flipped output can be obtained as well:
    >>> print(np.fliplr(convert_uint_array_to_binary(np.array([1, 2, 16, 2**16 - 1]))))  # doctest: +NORMALIZE_WHITESPACE
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]

    Or, via the provided argument 'bit_order':
    >>> print(convert_uint_array_to_binary(np.array([1, 2, 16, 2**16 - 1]), bit_order='big_first'))  # doctest: +NORMALIZE_WHITESPACE
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
        [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]

    Also note that other bit sizes are supported:
    >>> print(convert_uint_array_to_binary(np.array([1, 2, 7]), num_bits=3)) # doctest: +NORMALIZE_WHITESPACE
    [[1 0 0]
     [0 1 0]
     [1 1 1]]

    Also makes sure that the chosen num_bits can be used for the representation(8 can't be represented with 3 bits):
    >>> print(convert_uint_array_to_binary(np.array([8]), num_bits=3))
    Traceback (most recent call last):
    ...
    AssertionError: the accepted values for conversion must be in the range [0,7]!
    """

    assert num_bits >= 0, "Bit number must be non-negative!"
    assert (
        (uint_array <= (2**num_bits - 1)) & (uint_array >= 0)
    ).all(), (
        f"the accepted values for conversion must be in the range [0,{2**num_bits-1}]!"
    )

    # Realistically, can skip the next check to assure that int arrays in the right range are still accepted.
    # Thus, the user must verify that his signed int array has only non-negative values.
    # assert np.issubdtype(uint_array.dtype, np.unsignedinteger), "the array must be an unsigned one!"

    xshape = list(uint_array.shape)
    uint_array = uint_array.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=uint_array.dtype).reshape([1, num_bits])

    # P.S the direction is low_bit first by default
    result = (uint_array & mask).astype(bool).astype(int).reshape(xshape + [num_bits])
    if bit_order == "big_first":
        return np.fliplr(result)
    return result


def convert_binary_array_to_unint(
    binary_array: np.ndarray,
    bit_order: Literal["big_first", "small_first"] = "small_first",
) -> np.ndarray:
    """Convert an array of bits into an (unsigned) integer array.

    Args:
        binary_array (np.ndarray): an array with each row representing a single number to convert. Must be
            assosiated with the correct bit_order.
        bit_order (Literal['big_first','small_first'], optional): Bit order of numbers. Defaults to 'small_first'.

    Returns:
        np.ndarray: array of converted numbers with each number corresponding to each row.

    Sample Usage:
    >>> array=convert_uint_array_to_binary(np.array([1, 2, 7]), num_bits=3)
    >>> convert_binary_array_to_unint(array) # doctest: +NORMALIZE_WHITESPACE
    array([[1],
           [2],
           [7]])

    Same as before but with explicit bit_order argument
    >>> array=convert_uint_array_to_binary(np.array([1, 2, 16, 2**16 - 1]), bit_order='small_first')
    >>> convert_binary_array_to_unint(array, bit_order='small_first') # doctest: +NORMALIZE_WHITESPACE
    array([[    1],
           [    2],
           [   16],
           [65535]])

    Same as before but with explicit bit_order argument set to 'big_first
    >>> array=convert_uint_array_to_binary(np.array([1, 2, 16, 2**16 - 1]), bit_order='big_first')
    >>> convert_binary_array_to_unint(array, bit_order='big_first') # doctest: +NORMALIZE_WHITESPACE
    array([[    1],
           [    2],
           [   16],
           [65535]])
    """
    # note that we can rely on the shape of binary_array to determine num_bits, so no need to pass it
    if bit_order == "small_first":
        powers_of_two_array = np.array(
            [2] * binary_array.shape[1], dtype=np.int32
        ) ** np.arange(binary_array.shape[1])
    else:
        powers_of_two_array = np.array(
            [2] * binary_array.shape[1], dtype=np.int32
        ) ** np.arange(binary_array.shape[1] - 1, -1, -1)
    return (binary_array @ powers_of_two_array).reshape(-1, 1)

if __name__=="__main__":
    N: Final = 10  # visible layer size
    M: Final = 10  # hidden layer size

    RBM_instance = RBM(visible_layer_size=N, hidden_layer_size=M)
