import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        # MY CODE
        product = input @ self.weight.T
        if self.bias is not None:
            product += self.bias
        return product

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        # MY CODE
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        # MY CODE
        grad_weight = grad_output.T @ input
        self.grad_weight += grad_weight

        if self.bias is not None:
            grad_bias = grad_output.sum(axis=0)
            self.grad_bias += grad_bias

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.inference_k = None
        self.inference_b = None

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        # MY CODE

        if self.training:
            self.mean = input.mean(axis=0)
            self.var = input.var(axis=0)

            b = input.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * b / (b - 1) * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
        

        self.input_mean = input - self.mean
        self.sqrt_var = np.sqrt(self.var + self.eps)
        self.inv_sqrt_var = 1 / self.sqrt_var

        self.norm_input = self.input_mean * self.inv_sqrt_var

        y = self.norm_input 
        if self.affine:
            y = y * self.weight + self.bias
        
        return y

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        # MY CODE
        grad_input_norm = (grad_output * self.weight) if self.affine else grad_output

        if not self.training:
            return grad_input_norm * self.inv_sqrt_var
        
        b = input.shape[0]
        grad_var = (self.input_mean * grad_input_norm).sum(axis=0) * \
                    (self.inv_sqrt_var ** 3) * -0.5
        grad_mean = -1 * grad_input_norm.sum(axis=0) * self.inv_sqrt_var
        

        grad_input = grad_input_norm * self.inv_sqrt_var 
        grad_input += 2 / b * grad_var * self.input_mean
        grad_input += 1 / b * grad_mean

        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        # MY CODE
        
        if self.affine:
            self.grad_bias += grad_output.sum(axis=0)
            self.grad_weight += (self.norm_input * grad_output).sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'

    def train(self):
        super().train()
        self.inference_k = None
        self.inference_b = None
    
    def eval(self):
        super().eval()

        weight = self.weight if self.affine else 1
        bias = self.bias if self.affine else 0

        self.inference_k = weight / np.sqrt(self.running_var + self.eps)
        self.inference_b = bias - (self.inference_k * self.running_mean)


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # MY CODE
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
            return input * self.mask / (1 - self.p)
        else:
            return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # MY CODE
        if self.training:
            return grad_output * self.mask / (1 - self.p)
        else:
            return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        # MY CODE
        output = input
        self.inputs = []
        for module in self.modules:
            self.inputs.append(output)
            output = module(output)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        # MY CODE
        grad_input = grad_output
        for i in reversed(range(len(self.modules))):
            module = self.modules[i]
            input = self.inputs[i]
            grad_input = module.backward(input, grad_input)
        return grad_input

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
