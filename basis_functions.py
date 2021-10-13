# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and some common implementations."""

import abc

#抽象类的作用就是控制子类的方法的名称，要求子类必须按照父类的要求的实现指定的方法，且方法名要和父类保持一致
# 注意事项
# 1、抽象类不能被实例化
#
# 2、子类需要严格遵守父类的抽象类的规则，而孙类不需要遵守这个规则
#
# 3、如果想使用抽象类，则该类只需要继承抽象类即可

#有时，我们抽象出一个基类，知道要有哪些方法，但只是抽象方法，并不实现功能，只能继承，而不能被实例化，但子类必须要实现该方法，这就需要用到抽象基类，在很多时候用到的很多

import numpy as np

from functools import reduce
#reduce(function, iterable[, initializer])
#用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果


class BasisFunction(object):

    r"""ABC for basis functions used by LSPI Policies.

    A basis function is a function that takes in a state vector and an action
    index and returns a vector of features. The resulting feature vector is
    referred to as :math:`\phi` in the LSPI paper (pg 9 of the PDF referenced
    in this package's documentation). The :math:`\phi` vector is dotted with
    the weight vector of the Policy to calculate the Q-value.

    The dimensions of the state vector are usually much larger than the dimensions
    of the :math:`\phi` vector. However, the dimensions of the :math:`\phi`
    vector are usually much smaller than the dimensions of an exact
    representation of the state which leads to significant savings when
    computing and storing a policy.

    """

    __metaclass__ = abc.ABCMeta  # 使用的元类方法，元类是abc定义的一个工具类，抽象类通过元类来定义

    @abc.abstractmethod # 抽象类的抽象方法，抽象方法表示基类的一个方法，没有实现，所以基类不能实例化，子类实现了该抽象方法才能被实例化。
    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).

        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix for the given state-action pair.

        The way this value is calculated depends entirely on the concrete
        implementation of BasisFunction.

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.


        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        """
        pass  # pragma: no cover

    @abc.abstractproperty
    def num_actions(self):
        """Return number of possible actions.

        Returns
        -------
        int
            Number of possible actions.
        """
        pass  # pragma: no cover

    @staticmethod
    # 静态方法是类中的函数，不需要实例。静态方法主要是用来存放逻辑性的代码，主要是一些逻辑属于类，但是和类本身没有交互，
    # 即在静态方法中，不会涉及到类中的方法和属性的操作。可以理解为将静态方法存在此类的名称空间中。事实上，在python引入静态方法之前，通常是在全局名称空间中创建函数。
    def _validate_num_actions(num_actions):
        """Return num_actions if valid. Otherwise raise ValueError.

        Return
        ------
        int
            Number of possible actions.

        Raises
        ------
        ValueError
            If num_actions < 1

        """
        if num_actions < 1:
            raise ValueError('num_actions must be >= 1')
        return num_actions


class FakeBasis(BasisFunction):

    r"""Basis that ignores all input. Useful for random sampling.

    When creating a purely random Policy a basis function is still required.
    This basis function just returns a :math:`\phi` equal to [1.] for all
    inputs. It will however, still throw exceptions for impossible values like
    negative action indexes.

    """

    def __init__(self, num_actions):
        """Initialize FakeBasis."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

    def size(self):
        r"""Return size of 1.

        Returns
        -------
        int
            Size of :math:`phi` which is always 1 for FakeBasis

        Example
        -------

        >>> FakeBasis().size()
        1

        """
        return 1

    def evaluate(self, state, action):
        r"""Return :math:`\phi` equal to [1.].

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s. FakeBasis ignores these
            values.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a. FakeBasis ignores these
            values.

        Returns
        -------
        numpy.array
            :math:`\phi` vector equal to [1.].

        Raises
        ------
        IndexError
            If action index is < 0

        Example
        -------

        >>> FakeBasis().evaluate(np.arange(10), 0)
        array([ 1.])

        """
        if action < 0:
            raise IndexError('action index must be >= 0') # raise 抛出异常
        if action >= self.__num_actions:
            raise IndexError('action must be < num_actions')
        return np.array([1.])

    @property     # 把方法变成属性
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter # 把方法变成给属性赋值
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            If value < 1.

        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class OneDimensionalPolynomialBasis(BasisFunction):

    """Polynomial features for a state with one dimension.

    Takes the value of the state and constructs a vector proportional
    to the specified degree and number of actions. The polynomial is first
    constructed as [..., 1, value, value^2, ..., value^k, ...]
    where k is the degree. The rest of the vector is 0.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    num_actions: int
        The total number of possible actions

    Raises
    ------
    ValueError
        If degree is less than 0
    ValueError
        If num_actions is less than 1

    """

    def __init__(self, degree, num_actions):
        """Initialize polynomial basis function."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        if degree < 0:
            raise ValueError('Degree must be >= 0')
        self.degree = degree

    def size(self):
        """Calculate the size of the basis function.

        The base size will be degree + 1. This basic matrix is then
        duplicated once for every action. Therefore the size is equal to
        (degree + 1) * number of actions


        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.


        Example
        -------

        >>> basis = OneDimensionalPolynomialBasis(2, 2)
        >>> basis.size()
        6

        """
        return (self.degree + 1) * self.__num_actions

    def evaluate(self, state, action):
        r"""Calculate :math:`\phi` matrix for given state action pair.

        The :math:`\phi` matrix is used to calculate the Q function for the
        given policy.

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        Raises
        ------
        IndexError
            If :math:`0 \le action < num\_actions` then IndexError is raised.
        ValueError
            If the state vector has any number of dimensions other than 1 a
            ValueError is raised.

        Example
        -------

        >>> basis = OneDimensionalPolynomialBasis(2, 2)
        >>> basis.evaluate(np.array([2]), 0)
        array([ 1.,  2.,  4.,  0.,  0.,  0.])

        """
        if action < 0 or action >= self.__num_actions:
            raise IndexError('Action index out of bounds')

        if state.shape != (1, ):
            raise ValueError('This class only supports one dimensional states')

        phi = np.zeros((self.size(), ))

        offset = (self.size()/self.__num_actions)*action

        value = state[0]

        phi[offset:offset + self.degree + 1] = \
            np.array([pow(value, i) for i in range(self.degree+1)])

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            If value < 1.

        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class RadialBasisFunction(BasisFunction):

    r"""Gaussian Multidimensional Radial Basis Function (RBF).

    Given a set of k means :math:`(\mu_1 , \ldots, \mu_k)` produce a feature
    vector :math:`(1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
    e^{-\gamma || s - \mu_k ||^2})` where `s` is the state vector and
    :math:`\gamma` is a free parameter. This vector will be padded with
    0's on both sides proportional to the number of possible actions
    specified.

    Parameters
    ----------
    means: list(numpy.array)
        List of numpy arrays representing :math:`(\mu_1, \ldots, \mu_k)`.
        Each :math:`\mu` is a numpy array with dimensions matching the state
        vector this basis function will be used with. If the dimensions of each
        vector are not equal than an exception will be raised. If no means are
        specified then a ValueError will be raised
    gamma: float
        Free parameter which controls the size/spread of the Gaussian "bumps".
        This parameter is best selected via tuning through cross validation.
        gamma must be > 0.
    num_actions: int
        Number of actions. Must be in range [1, :math:`\infty`] otherwise
        an exception will be raised.

    Raises
    ------
    ValueError
        If means list is empty
    ValueError
        If dimensions of each mean vector do not match.
    ValueError
        If gamma is <= 0.
    ValueError
        If num_actions is less than 1.

    Note
    ----

    The numpy arrays specifying the means are not copied.

    """

    def __init__(self, means, gamma, num_actions):
        """Initialize RBF instance."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        if len(means) == 0:
            raise ValueError('You must specify at least one mean')

        if reduce(RadialBasisFunction.__check_mean_size, means) is None:
            raise ValueError('All mean vectors must have the same dimensions')

        self.means = means

        if gamma <= 0:
            raise ValueError('gamma must be > 0')

        self.gamma = gamma

    @staticmethod
    def __check_mean_size(left, right):
        """Apply f if the value is not None.

        This method is meant to be used with reduce. It will return either the
        right most numpy array or None if any of the array's had
        differing sizes. I wanted to use a Maybe monad here,
        but Python doesn't support that out of the box.

        Return
        ------
        None or numpy.array
            None values will propogate through the reduce automatically.

        """
        if left is None or right is None:
            return None
        else:
            if left.shape != right.shape:
                return None
        return right

    def size(self):
        r"""Calculate size of the :math:`\phi` matrix.

        The size is equal to the number of means + 1 times the number of
        number actions.

        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.

        """
        return (len(self.means) + 1) * self.__num_actions

    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix.

        Matrix will have the following form:

        :math:`[\cdots, 1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
        e^{-\gamma || s - \mu_k ||^2}, \cdots]`

        where the matrix will be padded with 0's on either side depending
        on the specified action index and the number of possible actions.

        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        Raises
        ------
        IndexError
            If :math:`0 \le action < num\_actions` then IndexError is raised.
        ValueError
            If the state vector has any number of dimensions other than 1 a
            ValueError is raised.

        """
        if action < 0 or action >= self.__num_actions:
            raise IndexError('Action index out of bounds')

        if state.shape != self.means[0].shape:
            raise ValueError('Dimensions of state must match '
                             'dimensions of means')

        phi = np.zeros((self.size(), ))
        offset = (len(self.means[0])+1)*action

        rbf = [RadialBasisFunction.__calc_basis_component(state,
                                                          mean,
                                                          self.gamma)
               for mean in self.means]
        phi[offset] = 1.
        phi[offset+1:offset+1+len(rbf)] = rbf

        return phi

    @staticmethod
    def __calc_basis_component(state, mean, gamma):
        mean_diff = state - mean
        return np.exp(-gamma*np.sum(mean_diff*mean_diff))

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            If value < 1.

        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class ExactBasis(BasisFunction):

    """Basis function with no functional approximation.

    This can only be used in domains with finite, discrete state-spaces. For
    example the Chain domain from the LSPI paper would work with this basis,
    but the inverted pendulum domain would not.

    Parameters
    ----------
    num_states: list
        A list containing integers representing the number of possible values
        for each state variable.
    num_actions: int
        Number of possible actions.
    """

    def __init__(self, num_states, num_actions):
        """Initialize ExactBasis."""
        if len(np.where(num_states <= 0)[0]) != 0:
            raise ValueError('num_states value\'s must be > 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self._num_states = num_states

        self._offsets = [1]
        for i in range(1, len(num_states)):
            self._offsets.append(self._offsets[-1]*num_states[i-1])

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return reduce(lambda x, y: x*y, self._num_states, 1) * self.__num_actions

    def get_state_action_index(self, state, action):
        """Return the non-zero index of the basis.

        Parameters
        ----------
        state: numpy.array
            The state to get the index for.
        action: int
            The state to get the index for.

        Returns
        -------
        int
            The non-zero index of the basis

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        """
        if action < 0:
            raise IndexError('action index must be >= 0')
        if action >= self.__num_actions:
            raise IndexError('action must be < num_actions')

        base = action * int(self.size() / self.__num_actions)

        offset = 0
        for i, value in enumerate(state):
            offset += self._offsets[i] * state[i]

        return base + offset

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a single non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """
        if len(state) != len(self._num_states):
            raise ValueError('Number of state variables must match '
                             + 'size of num_states.')
        if len(np.where(state < 0)[0]) != 0:
            raise ValueError('state cannot contain negative values.')
        for state_var, num_state_values in zip(state, self._num_states):
            if state_var >= num_state_values:
                raise ValueError('state values must be <= corresponding '
                                 + 'num_states value.')

        phi = np.zeros(self.size())
        phi[self.get_state_action_index(state, action)] = 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class OneshotBasis(BasisFunction):
    def __init__(self, num_states, num_actions):
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self.__num_states = num_states

    def size(self):
        return self.__num_actions * self.__num_states

    def evaluate(self, state, action):
        phi = np.zeros((self.size(),))
        phi[state + action * self.__num_states] = 1
        return phi

    @property     # 把方法变成属性
    def num_actions(self):
        return self.__num_actions

    @num_actions.setter # 把方法变成给属性赋值
    def num_actions(self, value):
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value



