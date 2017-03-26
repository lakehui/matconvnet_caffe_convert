function caffe_net = convertNet(caffe_model,  mat_weight,  savepath)
% caffe_model: the path of caffemodel.prototxt
% mat_weight: the path of matconv.mat
% savepath: the path of caffenet.caffemodel

    % convert a matconvnet DAG model to caffe model

    addpath('../../matlab');

    %mat_weight = '/home/hh/code/vggvein/veinDagnn/data/vein-baseline/net-12_openset_model.mat'
    mat_net = mat_weight;%load(mat_weight);
    % if matconvnet model is simpleNet, you should transform the simpleNet to DAG firstly
    %net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;


    % caffe_model = '/home/hh/Documents/caffe/models/convert_FVNet/FV-Net.prototxt';
    caffe_net = caffe.Net(caffe_model, 'test');

    for i = 1 : size(mat_net.layers, 2)
        mark_success = false;
        for j = 1 : size(caffe_net.layer_names, 1)

            if strcmp(mat_net.layers(i).name, caffe_net.layer_names{j})
                if ~strcmp(mat_net.layers(i).type, 'dagnn.BatchNorm')
                    for n_param = 1 : size(mat_net.layers(i).params, 2)
                        param_name = mat_net.layers(i).params(n_param);
                        param_blob = find_params(mat_net, param_name);
                        caffe_net.params(caffe_net.layer_names{j}, n_param).set_data(param_blob);
                    end
                    mark_success = true;
                    disp([mat_net.layers(i).name, ' layer in matconv -> caffe layer success '])
                end

                if strcmp(mat_net.layers(i).type, 'dagnn.BatchNorm')
                    %BN layer consist of batchnorm and scale layer in caffe
                    %you should indicate the epsion param  in BatchNorm layer,
                    %because the variance is store in caffe, but std in matconvnet                
                    param_name = mat_net.layers(i).params(1);
                    param_blob = find_params(mat_net, param_name);
                    caffe_net.params(caffe_net.layer_names{j+1}, 1).set_data(param_blob); % set scale layer params
                    param_name = mat_net.layers(i).params(2);
                    param_blob = find_params(mat_net, param_name);
                    caffe_net.params(caffe_net.layer_names{j+1}, 2).set_data(param_blob);

                    param_name = mat_net.layers(i).params(3);
                    param_blob = find_params(mat_net, param_name);
                    caffe_net.params(caffe_net.layer_names{j}, 1).set_data(param_blob(:,1));
                    caffe_net.params(caffe_net.layer_names{j}, 2).set_data(param_blob(:,2).^2);
                    caffe_net.params(caffe_net.layer_names{j}, 3).set_data(1.0);

                    mark_success = true;
                    disp([mat_net.layers(i).name, ' layer in matconv -> caffe layer success '])
                end
                break;
            end
        end
        if mark_success ~= true
            disp([mat_net.layers(i).name, ' layer of matconvnet is left'])
        end
    end
    
    % save caffe model to file
    caffe_net.save([savepath, '.caffemodel'])
end

function param_blob = find_params(mat_net, param_name)
    for i = 1 : size(mat_net.params, 2)
        if strcmp(mat_net.params(i).name, param_name)
            param_blob = mat_net.params(i).value;
            break;
        end
    end
end
