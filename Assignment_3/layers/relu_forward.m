function z = relu_forward(x)
    z = (x + abs(x))./2; 
end
