function dldx = relu_backward(x, dldz)
    dldx = dldz.*(x>0);
end
