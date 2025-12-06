import numpy as np
import pycuda.driver as cuda


class MatrixStruct(object):
    def __init__(self, array):
        self._cptr = None
        array = np.array(array, dtype=np.float32)
        self.shape, self.dtype = array.shape, array.dtype
        self.width = np.int32(self.shape[1])
        self.height = np.int32(self.shape[0])
        self.padding = np.int32(0)
        if len(self.shape) == 3:
            self.padding = np.int32(self.shape[0])
            self.width = np.int32(self.shape[2])
            self.height = np.int32(self.shape[1])
        self.stride = self.width
        self.elements = cuda.to_device(array)  # 分配内存并拷贝数组数据至device，返回其地址

    def send_to_gpu(self):
        self._cptr = cuda.mem_alloc(self.nbytes())  # 分配一个C结构体所占的内存
        cuda.memcpy_htod(int(self._cptr), self.width.tobytes())  # 拷贝数据至device，下同
        cuda.memcpy_htod(int(self._cptr) + 4, self.height.tobytes())
        cuda.memcpy_htod(int(self._cptr) + 8, self.stride.tobytes())
        cuda.memcpy_htod(int(self._cptr) + 12, self.padding.tobytes())
        cuda.memcpy_htod(int(self._cptr) + 16, np.intp(int(self.elements)).tobytes())

    def get_from_gpu(self, c_k):
        return cuda.from_device(self.elements, (c_k, self.shape[1]), self.dtype)  # 从device取回数组数据

    def get_from_gpu3(self):
        return cuda.from_device(self.elements, self.shape, self.dtype)  # 从device取回数组数据

    def nbytes(self):
        return self.width.nbytes * 4 + np.intp(0).nbytes

    def rehtod(self, arr):
        cuda.memcpy_htod(self.elements, arr)
