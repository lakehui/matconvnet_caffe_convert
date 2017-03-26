%% -------- caffe forward-----
addpath('../../matlab');

model = '/home/hh/Documents/caffe/models/convert_FVNet/FV-Net.prototxt';
weight = '/home/hh/Documents/caffe/models/convert_FVNet/FV-Net.caffemodel';

caffe_net = caffe.Net(model, weight, 'test');

caffe_net.forward({single(im2)});

caffe_out = caffe_net.blobs('conv6_1').get_data();

%% -------------- matconv forward-----------
run('matconvnet_root/matlab/vl_setupnn.m');

mat_net = dagnn.DagNN.loadobj(mat_net) ;

mat_net.mode = 'test';

inputs = {'input_1', single(im), 'labels', 1} ;

mat_net.eval(inputs) ;

mat_out = mat_net.vars(end).value;

%% 
if caffe_out == mat_out
    disp('convert success');
else
    disp('the last layer output is not equal');
end