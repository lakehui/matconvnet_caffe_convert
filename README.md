# matconvnet_caffe_convert
the project convert a matconvnet model to caffe model

# config requirement
MatConvNet,    caffe with matcaffe interface

# usage
you can run the function in matlab to get a caffe net which is converted from a matconvnet model;

    caffe_net = convertNet(caffe_model,  mat_weight,  savepath)
          
    output:
    
        caffe_net: the caffe net
        
    input:
    
        caffe_model: the file path of caffe model prototxt;
        
        mat_weight: the matconvnet model, the simmplenn structure and DAG structure both are allowed, but if the model is simplenn, you             should using "net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;" to obtain the DAG structure.
        
    savepath: 
    
        the savepath of caffe model
        
    
