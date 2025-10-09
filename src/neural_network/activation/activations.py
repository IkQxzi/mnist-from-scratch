import numpy as np 


class Activation():

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray: # type: ignore
        pass

    def fwd(self, inputs) -> None:
        self.outputs = self._fwd_func(inputs)

    def bwd(self) -> None:
        pass


class ReLU(Activation):

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray:
        outputs = np.maximum(inputs, 0)
        print(outputs)
        return outputs

relu = ReLU()
test_data = np.random.randn(4, 3)
print(test_data)
out = relu.fwd(test_data)
print(out)

