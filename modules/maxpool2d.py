from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
 def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

 def forward(self, input, training=True):  
    self.input = input   
    B, C, H, W = input.shape
    KH, KW = self.kernel_size, self.kernel_size
    SH, SW = self.stride, self.stride
    out_h = (H - KH) // SH + 1
    out_w = (W - KW) // SW + 1
   
    output = np.full((B, C, out_h, out_w), -np.inf, dtype=input.dtype)
    
    # Para cada posición (ki, kj) del kernel, comparar con TODA la ventana de salida a la vez
    for ki in range(KH):
        for kj in range(KW):
            # elementos en posición (ki, kj) 
            slice_view = input[:, :, ki:ki + out_h*SH:SH, kj:kj + out_w*SW:SW]
            # máximo elemento a elemento
            output = np.maximum(output, slice_view)
    
    return output
def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input