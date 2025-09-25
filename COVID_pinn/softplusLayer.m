classdef softplusLayer < nnet.layer.Layer
    methods
        function layer = softplusLayer(name)
            if nargin > 0
                layer.Name = name;
            end
            layer.Description = "Softplus activation layer (positive outputs)";
        end
        function Z = predict(~,X)
            Z = log(1 + exp(X));
        end
    end
end