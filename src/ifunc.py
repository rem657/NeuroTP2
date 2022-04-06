from typing import Union

import numpy as np


class IFunc:
	def __init__(self, name):
		self.name = name

	def __call__(self, t_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
		raise NotImplementedError()


class IConst(IFunc):
	def __init__(self, value: float, name=None):
		if name is None:
			name = f"IConst_v{str(value).replace('.', '_')}"
		super(IConst, self).__init__(name)
		self.value = value

	def __call__(self, t_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
		return self.value * np.ones_like(t_ms)


class ISteps(IFunc):
	def __init__(self, current_values: np.ndarray, step_len=20, inactive_len=20, alt=False, name=None):
		if name is None:
			name = "ISteps_alt" if alt else "ISteps"
		super(ISteps, self).__init__(name)
		self.current_values = current_values
		self.step_len = step_len
		self.inactive_len = inactive_len
		self.alt = alt
		self._sign = 1

	def __call__(self, t_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
		active = (t_ms % (self.inactive_len + self.step_len)) > self.inactive_len
		index = (t_ms // (self.inactive_len + self.step_len)).astype(int) % len(self.current_values)
		values = self.current_values[index]*active + (-self.current_values[index] if self.alt else 0.0) * (1-active)
		return values


class ISin(IFunc):
	def __init__(self, period: float, amplitude: float = 1.0):
		super(ISin, self).__init__(f"ISin_p{str(period).replace('.', '_')}_a{str(amplitude).replace('.', '_')}")
		self.period = period
		self.amplitude = amplitude

	def __call__(self, t_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
		return self.amplitude*np.sin(2*np.pi*t_ms / self.period)





