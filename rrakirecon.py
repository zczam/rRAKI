# © 2021 Regents of the University of Minnesota. Academic and Research Use Only.

#import print_function
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import matplotlib.pyplot as plt


def weight_variable(shape,vari_name):                   # notice here, we can reuse those weight which gives fast convergence or the result from last iteration instead of using new random weight.
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape,dtype=tf.float32)
    return tf.Variable(initial,name='Bias')


def ZcCeil(input):
    return np.int32(np.ceil(input))

# add cnn layer only adds cnn layer. activation is independent from this function for better execution performance.
def ZcAddCNNlayer_Dilate(inputTensor,weight_shape,dilate_rate_local,padding_option):

    weight_local = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1,dtype=tf.float32),name='unbiased_convlayer') # initialize the weight
    conv_result = tf.nn.convolution(inputTensor, weight_local,padding=padding_option,dilation_rate = [1,dilate_rate_local]); # convolution with dilation.

    return [conv_result, weight_local] # returning resulted convolution results, the weights for this layer.

def ZcAddCNNlayer_Dilate_Biased(inputTensor,weight_shape,dilate_rate_local,padding_option):

    weight_local = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1,dtype=tf.float32),name='biased_convLayer') # initialize the weight
    conv_result = tf.nn.convolution(inputTensor, weight_local,padding=padding_option,dilation_rate = [1,dilate_rate_local]); # convolution with dilation.
    bias_local = bias_variable([weight_shape[3]])
    conv_result = tf.nn.bias_add(conv_result,bias_local)
    return [conv_result, weight_local, bias_local] # returning resulted convolution results, the weights for this layer.

def ZcAddCNN_dense_layer_Dilate(inputTensor,weight_shape,dilate_rate_local,padding_option):

    weight_local = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1,dtype=tf.float32)) # initialize the weight
    conv_result = tf.concat([inputTensor, tf.nn.convolution(inputTensor, weight_local,padding=padding_option,dilation_rate = [1,dilate_rate_local])], axis=3); # convolution with dilation.

    return [conv_result, weight_local] # returning resulted convolution results, the weights for this layer.


# add trainable bias to a 3D tensor
def ZcAddBias3Dtensor(inputTensor):

    bias_local = tf.Variable(tf.constant(0.1, shape=[1,1,inputTensor.shape(2)],dtype=tf.float32)) # initialize the bias
    return [inputTensor + bias_local, bias_local] # returning resulted convolution results, the weights for this layer.


def ZcAddCNNlayer_Dilate_Conti(inputTensor,weight,dilate_rate_local,padding_option):
    conv_result = tf.nn.convolution(inputTensor, tf.Variable(weight,name='unbiased_convlayer',trainable=False),padding=padding_option,dilation_rate = [1,dilate_rate_local]); # convolution with dilation.
    return conv_result # returning resulted convolution results, the weights for this layer.

def ZcAddCNNlayer_Dilate_Biased_Conti(inputTensor,weight,bias,dilate_rate_local,padding_option):
    conv_result = tf.nn.bias_add(tf.nn.convolution(inputTensor, tf.Variable(weight,name='biased_convLayer',trainable=False) ,padding=padding_option,dilation_rate = [1,dilate_rate_local]),tf.Variable(bias,trainable=False))
    return conv_result # returning resulted convolution results, the weights for this layer.





def ZcSaveMassiveWeights(rRAKI_weights_all_values,Route,no_channels,layer_amount,layer_amount2,layer_amount3):

    if not (os.path.exists(Route)):
        os.mkdir(Route)

    for ind_shift_Channel_pos in range(no_channels):

       
        for layer_idx in range(layer_amount):
            data = np.float32(rRAKI_weights_all_values[(('linear_weights_layer%d_channel%d') % ((layer_idx+1),ind_shift_Channel_pos))])
            name = Route+(('/linear_weights_layer%d_channel%d') % (layer_idx,ind_shift_Channel_pos)) + '.mat'
            sio.savemat(name,{'data':data})  # save the results      
       
        for layer_idx in range(layer_amount):
            data = np.float32(rRAKI_weights_all_values[(('nonlinear_weights_layer%d_F_Low_channel%d') % ((layer_idx+1),ind_shift_Channel_pos))])
            name = Route+(('/nonlinear_weights_layer%d_F_Low_channel%d') % (layer_idx,ind_shift_Channel_pos)) + '.mat'
            sio.savemat(name,{'data':data})  # save the results      
       
       
           

def ZcLoadMassiveWeights(Route,no_channels,layer_amount,layer_amount2,layer_amount3):
    
    if (os.path.exists(Route)):
        rRAKI_weights_all_values2 = {}
        
        for ind_shift_Channel_pos in range(no_channels):

            for layer_idx in range(layer_amount):
                name = Route+(('/linear_weights_layer%d_channel%d') % (layer_idx,ind_shift_Channel_pos)) + '.mat'
                file = sio.loadmat(name)
                data = file['data']
                rRAKI_weights_all_values2[(('linear_weights_layer%d_channel%d') % ((layer_idx+1),ind_shift_Channel_pos))] = data    
            
            for layer_idx in range(layer_amount3):     
                name = Route+(('/nonlinear_weights_layer%d_F_trans_channel%d') % (layer_idx,ind_shift_Channel_pos)) + '.mat'
                file = sio.loadmat(name)
                data = file['data']
                rRAKI_weights_all_values2[(('nonlinear_weights_layer%d_F_trans_channel%d') % ((layer_idx+1),ind_shift_Channel_pos))] = data       
            

    return rRAKI_weights_all_values2


# this function gives the valid area index needed in RAKI/GRAPPA.
# input tensor is 3D with batch dimension, [batch,x,y,z]

# input tensor is 3D, [x,y,z]
def ZcGetValidArea3D_actual2(inputTensor):

    [dim2,dim3,dim4]=np.shape(inputTensor)
    for x_idx in range(dim2):
        if(np.sum(np.abs(inputTensor[x_idx,:,ZcCeil(dim4/2)]))!=0):
            x_start = x_idx
            break

    for x_idx in range(dim2-1,-1,-1):
        if(np.sum(np.abs(inputTensor[x_idx,:,ZcCeil(dim4/2)]))!=0):
            x_end = x_idx
            break

    for y_idx in range(dim3):
        if(np.sum(np.abs(inputTensor[:,y_idx,ZcCeil(dim4/2)]))!=0):
            y_start = y_idx
            break

    for y_idx in range(dim3-1,-1,-1):
        if(np.sum(np.abs(inputTensor[:,y_idx,ZcCeil(dim4/2)]))!=0):
            y_end = y_idx
            break
       
       
    return [x_start, x_end, y_start, y_end]




# this function finds the
def ZcGetValidArea3D_analytical_convolution_FinalResultSize(inputTensor,linear_weight_shape,weight_shape_list,acc_rate):
   
    [dim1,layer_amount] = np.int32(np.shape(weight_shape_list));
    [dim0,dimX,dimY,dimZ] = np.int32(np.shape(inputTensor))
    y_start_linear = 0
    y_end_linear = dimY - 1;
    x_start_linear = 0
    x_end_linear = dimX - 1;
   
    y_start_nonlinear = 0
    y_end_nonlinear = dimY - 1;
    x_start_nonlinear = 0
    x_end_nonlinear = dimX - 1;
   
    for layer_idx in range(layer_amount):
       
        if (weight_shape_list[0,layer_idx]>1):
            x_start_nonlinear = x_start_nonlinear + np.int32(np.floor(weight_shape_list[0,layer_idx]/2))
            x_end_nonlinear   = x_end_nonlinear - np.int32(np.floor(weight_shape_list[0,layer_idx]/2))
           
        if (weight_shape_list[1,layer_idx]>1):
            y_start_nonlinear = y_start_nonlinear + np.int32((np.floor((weight_shape_list[1,layer_idx])/2)-1)*acc_rate)
            y_end_nonlinear = y_end_nonlinear - np.int32(np.ceil((weight_shape_list[1,layer_idx])/2)*acc_rate)
   
    x_start_linear = np.int32(np.floor(linear_weight_shape[0]/2))
    x_end_linear   = x_end_linear - np.int32(np.floor(linear_weight_shape[0]/2))
   
    y_start_linear = np.int32((np.floor((linear_weight_shape[1])/2)-1)*acc_rate)
    y_end_linear = y_end_linear - np.int32((np.ceil(linear_weight_shape[1])/2)*acc_rate)
   
    if x_start_linear > x_start_nonlinear:
        x_start = x_start_linear
    else:
        x_start = x_start_nonlinear
       
    if x_end_linear < x_end_nonlinear:
        x_end = x_end_linear
    else:
        x_end = x_end_nonlinear

    if y_start_linear > y_start_nonlinear:
        y_start = y_start_linear
    else:
        y_start = y_start_nonlinear
       
    if y_end_linear < y_end_nonlinear:
        y_end = y_end_linear
    else:
        y_end = y_end_nonlinear

   
    return np.int32([x_start, x_end, y_start, y_end])


def ZcGetValidArea3D_analytical_2ndNetwork(inputTensor,linear_weight_shape,weight_shape_list,acc_rate):
    # inputTensor is in the original size
    [dim1,layer_amount] = np.int32(np.shape(weight_shape_list));
    [dim0,dimX,dimY,dimZ] = np.int32(np.shape(inputTensor))
   

    y_start_nonlinear = 0
    y_end_nonlinear = dimY - 1;
    x_start_nonlinear = 0
    x_end_nonlinear = dimX - 1;
   
   
    for layer_idx in range(layer_amount):
       
        if (weight_shape_list[0,layer_idx]>1):
            x_start_nonlinear = x_start_nonlinear + np.int32(np.floor(weight_shape_list[0,layer_idx]/2))
            x_end_nonlinear   = x_end_nonlinear - np.int32(np.floor(weight_shape_list[0,layer_idx]/2))
           
        if (weight_shape_list[1,layer_idx]>1):
            y_start_nonlinear = y_start_nonlinear + np.int32(np.floor((weight_shape_list[1,layer_idx])/2*acc_rate))
            y_end_nonlinear = y_end_nonlinear - np.int32(np.floor((weight_shape_list[1,layer_idx])/2*acc_rate))    
   

   
    return np.int32([x_start_nonlinear,x_end_nonlinear,y_start_nonlinear,y_end_nonlinear])



def ZcGetValidArea3D_analytical(inputTensor,linear_weight_shape,weight_shape_list,acc_rate):
    # inputTensor is in the original size
    [dim1,layer_amount] = np.int32(np.shape(weight_shape_list));
    [dim0,dimX,dimY,dimZ] = np.int32(np.shape(inputTensor))
    y_start_linear = 0
    y_end_linear = dimY - 1;
    x_start_linear = 0
    x_end_linear = dimX - 1;
   
    y_start_nonlinear = 0
    y_end_nonlinear = dimY - 1;
    x_start_nonlinear = 0
    x_end_nonlinear = dimX - 1;
   
    for layer_idx in range(layer_amount):
       
        if (weight_shape_list[0,layer_idx]>1):
            x_start_nonlinear = x_start_nonlinear + np.int32(np.floor(weight_shape_list[0,layer_idx]/2))
            x_end_nonlinear   = x_end_nonlinear - np.int32(np.floor(weight_shape_list[0,layer_idx]/2))
           
        if (weight_shape_list[1,layer_idx]>1):
            y_start_nonlinear = y_start_nonlinear + np.int32(np.floor((weight_shape_list[1,layer_idx]-1)*acc_rate/2))
            y_end_nonlinear = y_end_nonlinear - np.int32(np.ceil((weight_shape_list[1,layer_idx]-1)*acc_rate/2))
   
    x_start_linear = np.int32(np.floor(linear_weight_shape[0]/2))
    x_end_linear   = dimX - 1 - np.int32(np.floor(linear_weight_shape[0]/2))
   
    y_start_linear = np.int32(np.floor((linear_weight_shape[1]-1)*acc_rate/2))
    y_end_linear = dimY - 1 - np.int32(np.ceil((linear_weight_shape[1]-1)*acc_rate/2))
   
    diff_blockx =  np.floor(np.abs((x_end_nonlinear - x_start_nonlinear) - (x_end_linear - x_start_linear))/2)
    diff_blocky_S =  np.floor(np.abs((y_end_nonlinear - y_start_nonlinear) - (y_end_linear - y_start_linear))/acc_rate/2) # start has less one block than end
    diff_blocky_E =  np.ceil(np.abs((y_end_nonlinear - y_start_nonlinear) - (y_end_linear - y_start_linear))/acc_rate/2)    
   
    # find a smaller region covered by linear/nonlinear part as final valid area
    if (x_end_linear - x_start_linear) < (x_end_nonlinear - x_start_nonlinear):
        # linear part has smaller region along x
        x_end_nonlinear = x_end_nonlinear - diff_blockx;
        x_start_nonlinear = x_start_nonlinear + diff_blockx;
    else:
        # nonlinear part has smaller region along x
        x_end_linear = x_end_linear - diff_blockx;
        x_start_linear = x_start_linear + diff_blockx;
       
    if (y_end_linear - y_start_linear) < (y_end_nonlinear - y_start_nonlinear):
        # linear part has smaller region along y
        y_end_nonlinear = y_end_nonlinear - diff_blocky_E * acc_rate;
        y_start_nonlinear = y_start_nonlinear + diff_blocky_S * acc_rate;
    else:
        # nonlinear part has smaller region along y
        y_end_linear = y_end_linear - diff_blocky_E*acc_rate;
        y_start_linear = y_start_linear + diff_blocky_S*acc_rate;
   
    return np.int32([x_start_linear,x_end_linear,y_start_linear,y_end_linear,x_start_nonlinear,x_end_nonlinear,y_start_nonlinear,y_end_nonlinear])


# this function gives the energy map, respect to the complex energy.
# it outputs the same dimension as input
def ZcGetComplexEnergyFromMappedRealData(inputTensor):
    [dim1,dimX,dimY,dimChannel] = np.shape(inputTensor)
    EnergyMap = np.square(inputTensor[:,:,:,0:np.int32(dimChannel/2)]) + np.square(inputTensor[:,:,:,np.int32(dimChannel/2):])
    EnergyMap = np.reshape(EnergyMap,[dim1,dimX,dimY,np.int32(dimChannel/2)])
    EnergyMap = np.tile(EnergyMap,[1,1,1,2]);
    return EnergyMap


# just re-write the network function here
# for convinience
def ZcBuildNetwork_rRAKI_SingleChannel(inputTensor, target, linear_weight_shape, weight_shape_list,weight_shape_list2, acc_rate, sess):

    # initialize everything
    padding_option = 'SAME'
    [dim1,layer_amount] = np.shape(weight_shape_list[:,:]);
    [dim1,layer_amount2] = np.shape(weight_shape_list2[:,:]);
    target_inGraph = tf.placeholder(tf.float32, np.shape(target))  
    [x_start_linear,x_end_linear,y_start_linear,y_end_linear,x_start_nonlinear,x_end_nonlinear,y_start_nonlinear,y_end_nonlinear]= ZcGetValidArea3D_analytical(inputTensor,linear_weight_shape,np.squeeze(weight_shape_list[:,:]),acc_rate)

    # then define the network structure
   
    # Get Energy Map
    #amp_hi = np.max(abs(EnergyMap_singleChannel[:]))

    # a starting point of the threshold, set it as 0.01 of maximum of the ACS amplitude.
    #Energy_threshold = tf.Variable((amp_hi)*0.01)
   
    # linear part:
    #[linear_part, linear_weights] = ZcAddCNNlayer_Dilate(inputTensor,linear_weight_shape,acc_rate,padding_option)
    #linear_part_valid = linear_part[:,x_start_linear:x_end_linear+1,y_start_linear:y_end_linear+1,:]
    # non-linear part
    weights_list = {};
    print(np.shape(inputTensor)) 
   
    linear_part = np.copy(inputTensor);    
    for layer_idx in range(layer_amount):
        [linear_part, weight_this_layer] = ZcAddCNNlayer_Dilate(linear_part,np.squeeze(weight_shape_list[:,layer_idx]),acc_rate,padding_option)

        # gather the weights in to dictionary. here indexing begins from 1.
        weights_list[('weights_linear_layer%d' % (layer_idx+1))] = weight_this_layer        

    linear_part_valid = linear_part[:,x_start_nonlinear:x_end_nonlinear+1,y_start_nonlinear:y_end_nonlinear+1,:]
    #nonLinear_part_Denoise = tf.multiply(tf.cast(EnergyMap_singleChannel<Energy_threshold,dtype=tf.float32),nonLinear_part_Denoise)

   
   
    nonLinear_part_Denoise = np.copy(inputTensor);    
    for layer_idx in range(layer_amount):

        if((layer_idx<layer_amount-1) and layer_idx!=0):
            # CNN layer with dense connection
            [nonLinear_part_Denoise, weight_this_layer] = ZcAddCNNlayer_Dilate(nonLinear_part_Denoise,np.squeeze(weight_shape_list[:,layer_idx]),acc_rate,padding_option)
            # activation
            nonLinear_part_Denoise = tf.nn.leaky_relu(nonLinear_part_Denoise)
        else:
            # regular CNN layer as output
            [nonLinear_part_Denoise, weight_this_layer] = ZcAddCNNlayer_Dilate(nonLinear_part_Denoise,np.squeeze(weight_shape_list[:,layer_idx]),acc_rate,padding_option)

            if (layer_idx != (layer_amount-1)) :
                nonLinear_part_Denoise = tf.nn.leaky_relu(nonLinear_part_Denoise)

        # gather the weights in to dictionary. here indexing begins from 1.
        weights_list[('weights_layer%d_F_Low' % (layer_idx+1))] = weight_this_layer        

    nonLinear_part_Denoise_valid = nonLinear_part_Denoise[:,x_start_nonlinear:x_end_nonlinear+1,y_start_nonlinear:y_end_nonlinear+1,:]
    #nonLinear_part_Denoise = tf.multiply(tf.cast(EnergyMap_singleChannel<Energy_threshold,dtype=tf.float32),nonLinear_part_Denoise)

    [x_start_nonlinear2,x_end_nonlinear2,y_start_nonlinear2,y_end_nonlinear2] = ZcGetValidArea3D_analytical_2ndNetwork(linear_part_valid,linear_weight_shape,np.squeeze(weight_shape_list2[:,:]),acc_rate)  
    nonLinear_part_DeArtifact = tf.identity(nonLinear_part_Denoise_valid + linear_part_valid);
    for layer_idx in range(layer_amount2):

        if((layer_idx<layer_amount2-1) and layer_idx!=0):
            # CNN layer with dense connection
            [nonLinear_part_DeArtifact, weight_this_layer,bias_this_layer] = ZcAddCNNlayer_Dilate_Biased(nonLinear_part_DeArtifact,np.squeeze(weight_shape_list2[:,layer_idx]),acc_rate,padding_option)
            # activation
            nonLinear_part_DeArtifact = tf.nn.leaky_relu(nonLinear_part_DeArtifact)
        else:
            # regular CNN layer as output
            [nonLinear_part_DeArtifact, weight_this_layer,bias_this_layer] = ZcAddCNNlayer_Dilate_Biased(nonLinear_part_DeArtifact,np.squeeze(weight_shape_list2[:,layer_idx]),acc_rate,padding_option)

            if (layer_idx != (layer_amount2-1)) :
                nonLinear_part_DeArtifact = tf.nn.leaky_relu(nonLinear_part_DeArtifact)

        # gather the weights in to dictionary. here indexing begins from 1.
        weights_list[('weights_layer%d_F_High' % (layer_idx+1))] = weight_this_layer
        weights_list[('bias_layer%d_F_High' % (layer_idx+1))] = bias_this_layer
   
    nonLinear_part_DeArtifact_valid = nonLinear_part_DeArtifact[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:]
    #nonLinear_part_DeArtifact = tf.multiply(tf.cast(EnergyMap_singleChannel>=Energy_threshold,dtype=tf.float32),nonLinear_part_DeArtifact)
    nonLinear_part_Denoise_valid = nonLinear_part_Denoise_valid[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:]
    linear_part_valid = linear_part_valid[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:]
   
    target_valid = target_inGraph[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:]
    # grab valid index
    # linear part has less shrinkage than non-linear
    # here use VALID padding for linear part, then shrink the linear part from the 2nd non-linear weight layer
    # then we got the same size.

    # set loss
    norm_1o16 = tf.norm(target_valid[:,:,::acc_rate,:],ord=1)
    norm_1o16_l2= tf.norm(target_valid[:,:,::acc_rate,:])
 
    norm_all = tf.norm(target_valid[:,:,:,:],ord=1)

    lossFinal = tf.norm(target_valid - nonLinear_part_DeArtifact_valid - linear_part_valid - nonLinear_part_Denoise_valid,ord=1)/norm_all
    loss = lossFinal + tf.norm(target_valid[:,:,::acc_rate,:] - linear_part_valid[:,:,::acc_rate,:] - nonLinear_part_Denoise_valid[:,:,::acc_rate,:],ord=1)/norm_1o16
    lossG = tf.norm(target_valid[:,:,::acc_rate,:] - linear_part_valid[:,:,::acc_rate,:])/norm_1o16_l2
    loss = lossG + loss
    trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    g_vars = [var for var in trainable_collection if 'unbiased' in var.name]
    # set optimizer
    train_step = tf.train.AdamOptimizer(6e-4).minimize(loss)
    #train_stepG = tf.train.AdamOptimizer(6e-4).minimize(lossG)
    #train_step3 = tf.train.AdamOptimizer(2e-4).minimize(loss)
   
    variables_to_remove = list()
   
    print('Updating the following variables for total loss:')
    for vari in trainable_collection:
    #uses the attribute 'name' of the variable
        print(vari.name)

           
    for gvar in g_vars:
        variables_to_remove.append(gvar)
       
    for rem in variables_to_remove:
        trainable_collection.remove(rem)
   
   
    print('Updating the following variables for artifacts removal:')
    trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
   
    for vari in trainable_collection:
    #uses the attribute 'name' of the variable
        print(vari.name)
       
    train_step2= tf.train.AdamOptimizer(2e-4).minimize(lossFinal)


    init = tf.global_variables_initializer()
    sess.run(init)
    '''
    for i in range(1001):
        sess.run(train_stepG,feed_dict={target_inGraph: target})
        if i % 100 ==0:
            error_now=sess.run(lossG,feed_dict={target_inGraph: target})
            print('The',i,'th iteration gives an error',error_now)
    '''
    # run!
    for i in range(IterAmount_rRAKI):
        sess.run(train_step,feed_dict={target_inGraph: target})
        if i % 100 == 0:                                                                        
            error_now=sess.run(loss,feed_dict={target_inGraph: target})
            print('The',i,'th iteration gives an error',error_now)                            

    train_step.learning_rate=2e-4
    for i in range(1001):
        sess.run(train_step,feed_dict={target_inGraph: target})
        if i % 100 == 0:
            error_now=sess.run(loss,feed_dict={target_inGraph: target})
            print('The',i+IterAmount_rRAKI,'th iteration gives an error',error_now)
 

    for i in range(IterAmount_rRAKI2):
        sess.run(train_step2,feed_dict={target_inGraph: target})
        if i % 100 == 0:                                                                        
            error_now=sess.run(lossFinal,feed_dict={target_inGraph: target})
            print('The',i,'th iteration gives an error',error_now)                            

    # return the weights, in tf. gotta need a sess.run to grab them.
    return [weights_list]



def ZcTrain_rRAKI_Weights(trainingData,acc_rate,trainingflag):

    [dim1,dimX,dimY,dimZ] = np.shape(trainingData)
    no_channels = dimZ
    layer_amount = 5;
    # network settings here
    baseFrame = 1
    # linear part weights
    linear_weight_shape = [7,6,no_channels,acc_rate-1]
    # first, build up the weight list
    weight_shape_list = np.zeros([4,layer_amount],dtype=np.int32)

    # first layer from no_channels to blah
    weight_shape_list[:,0] = [7,6,no_channels,32];
    layer_accumulation = weight_shape_list[3,0]
    
    # 2nd to end-1 layers, 
    
    weight_shape_list[:,1] = [1,1,layer_accumulation, 64];
    layer_accumulation = weight_shape_list[3,1]
    
    weight_shape_list[:,2] = [3,2,layer_accumulation,32];  
    layer_accumulation = weight_shape_list[3,2]
    
    weight_shape_list[:,3] = [1,1,layer_accumulation,64];
    layer_accumulation = weight_shape_list[3,3]
    '''
    weight_shape_list[:,4] = [3,2,layer_accumulation,32];
    layer_accumulation = weight_shape_list[3,2]

    weight_shape_list[:,5] = [1,1,layer_accumulation,64];
    layer_accumulation = weight_shape_list[3,3]
    '''
    
    # the last layer
    weight_shape_list[:,layer_amount-1] = [5,4,layer_accumulation,acc_rate-1];
    
    
    layer_amount2 = 5;
    # first layer from no_channels to blah
    weight_shape_list2 = np.zeros([4,layer_amount2],dtype=np.int32)
    
    weight_shape_list2[:,0] = [5,1,acc_rate-1,(acc_rate-1)*4];
    layer_accumulation = weight_shape_list2[3,0]
    
    # 2nd to end-1 layers, 
    
    weight_shape_list2[:,1] = [1,1,layer_accumulation, (acc_rate-1)*4];
    layer_accumulation = weight_shape_list2[3,1]
    
    weight_shape_list2[:,2] = [3,1,layer_accumulation,(acc_rate-1)*4];  
    layer_accumulation = weight_shape_list2[3,2]
    
    weight_shape_list2[:,3] = [1,1,layer_accumulation,(acc_rate-1)*4];
    layer_accumulation = weight_shape_list2[3,3]
    
    # the last layer
    weight_shape_list2[:,layer_amount2-1] = [3,1,layer_accumulation,acc_rate-1];

    
    # Trans Network
    layer_amount3 = np.copy(layer_amount)
    weight_shape_list3 = np.copy(weight_shape_list);
    #weight_shape_list3[2,0] = acc_rate-1
    
    
    
    [x_start_linear,x_end_linear,y_start_linear,y_end_linear,x_start_nonlinear,x_end_nonlinear,y_start_nonlinear,y_end_nonlinear]= ZcGetValidArea3D_analytical(trainingData,linear_weight_shape,np.squeeze(weight_shape_list[:,:]),acc_rate)  
    [x_start_nonlinear2,x_end_nonlinear2,y_start_nonlinear2,y_end_nonlinear2] = ZcGetValidArea3D_analytical_2ndNetwork(trainingData,linear_weight_shape,np.squeeze(weight_shape_list2[:,:]),acc_rate)  
    [x_start,x_end,y_start,y_end] = ZcGetValidArea3D_analytical_convolution_FinalResultSize(trainingData,linear_weight_shape,weight_shape_list,acc_rate)
    
    rRAKI_weights_all_values = {}; # initialize the weight values storage


    return [rRAKI_weights_all_values,  linear_weight_shape, weight_shape_list,weight_shape_list2,weight_shape_list3, layer_amount,layer_amount2,layer_amount3]






# this function do the rRAKI to undersampled k-space with pre-trained weights
# inputKspace must be undersampled & zero-filled.
# the first line and the last line are sampled, this is ensured in pre-processing.

def ZcRecon_rRAKI(inputKspace, rRAKI_weights_all_values, linear_weight_shape, weight_shape_list,weight_shape_list2, acc_rate, layer_amount,layer_amount2):

    [dim1,dimX,dimY,dimZ] = np.shape(inputKspace)
    no_channels = dimZ
    padding_option = 'SAME'
    nonLinear_part_Denoise_recon = (np.zeros(np.shape(inputKspace)))
    nonLinear_part_DeArtifact_recon = (np.zeros(np.shape(inputKspace)))
    linear_recon = (np.copy(inputKspace))
    #inputKspace = (inputKspace)

    # in SMS, the last line of MB data is not sampled, we have more R-1 lines. so for rescon stage remove the ending R-1lines.
    '''
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.42))
    sess = tf.Session(config=config)  # tensorflow initialize
    init = tf.global_variables_initializer()
    sess.run(init)
    '''

    with tf.device('/gpu:0'):
        for ind_shift_Channel_pos in range(no_channels):
           
            [x_start_linear,x_end_linear,y_start_linear,y_end_linear,x_start_nonlinear,x_end_nonlinear,y_start_nonlinear,y_end_nonlinear]= ZcGetValidArea3D_analytical(inputKspace,linear_weight_shape,weight_shape_list[:,:],acc_rate)  
            [x_start,x_end,y_start,y_end] = ZcGetValidArea3D_analytical_convolution_FinalResultSize(inputKspace,linear_weight_shape,weight_shape_list,acc_rate)
            '''
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.42))
            sess = tf.Session(config=config)  # tensorflow initialize
            init = tf.global_variables_initializer()
            sess.run(init)
            '''
            # linear part
            #linear_weights = rRAKI_weights_all_values[(('linear_weights_channel%d') % ind_shift_Channel_pos)]
            #linear_part =  tf.nn.convolution(inputKspace, linear_weights, padding=padding_option, dilation_rate = [1,acc_rate])
            #linear_part_valid = sess.run(linear_part[:,x_start_linear:x_end_linear+1,y_start_linear:y_end_linear+1,:])      
            # non-linear part
   
            linear_part = tf.identity(inputKspace);    
            for layer_idx in range(layer_amount):
                convolutionWeights = tf.constant(rRAKI_weights_all_values[(('linear_weights_layer%d_channel%d')%((layer_idx+1),ind_shift_Channel_pos))])
                linear_part = tf.nn.convolution(linear_part, convolutionWeights,padding=padding_option,dilation_rate = [1,acc_rate]); # convolution with dilation.                # activation
           
            linear_part_valid = (linear_part[:,x_start_nonlinear:x_end_nonlinear+1,y_start_nonlinear:y_end_nonlinear+1,:])
   
           
            nonLinear_part_Denoise = tf.identity(inputKspace);    
            for layer_idx in range(layer_amount):
                convolutionWeights = tf.constant(rRAKI_weights_all_values[(('nonlinear_weights_layer%d_F_Low_channel%d')%((layer_idx+1),ind_shift_Channel_pos))])
                if((layer_idx<layer_amount-1) and layer_idx!=0):
                    # CNN layer with dense connection
                    nonLinear_part_Denoise = tf.nn.convolution(nonLinear_part_Denoise, convolutionWeights,padding=padding_option,dilation_rate = [1,acc_rate]); # convolution with dilation.                # activation
                    nonLinear_part_Denoise = tf.nn.leaky_relu(nonLinear_part_Denoise)
                else:
                    # regular CNN layer as output
                    nonLinear_part_Denoise = tf.nn.convolution(nonLinear_part_Denoise, convolutionWeights,padding=padding_option,dilation_rate = [1,acc_rate]); # convolution with dilation.                # activation
       
                    if (layer_idx != (layer_amount-1)) :
                        nonLinear_part_Denoise = tf.nn.leaky_relu(nonLinear_part_Denoise)    
       
            nonLinear_part_Denoise_valid = (nonLinear_part_Denoise[:,x_start_nonlinear:x_end_nonlinear+1,y_start_nonlinear:y_end_nonlinear+1,:])
   
            [x_start_nonlinear2,x_end_nonlinear2,y_start_nonlinear2,y_end_nonlinear2] = ZcGetValidArea3D_analytical_2ndNetwork(linear_part_valid,linear_weight_shape,np.squeeze(weight_shape_list2[:,:]),acc_rate)          
            nonLinear_part_DeArtifact = tf.identity(linear_part_valid+nonLinear_part_Denoise_valid);
            for layer_idx in range(layer_amount2):
                convolutionWeights = tf.constant(rRAKI_weights_all_values[(('nonlinear_weights_layer%d_F_High_channel%d')%((layer_idx+1),ind_shift_Channel_pos))])
                convolutionBias = tf.constant(np.squeeze(rRAKI_weights_all_values[(('nonlinear_bias_layer%d_F_High_channel%d')%((layer_idx+1),ind_shift_Channel_pos))]))
               
                if((layer_idx < layer_amount2-1) and layer_idx!=0):
                    # CNN layer with dense connection
                    nonLinear_part_DeArtifact = tf.nn.bias_add(tf.nn.convolution(nonLinear_part_DeArtifact, convolutionWeights,padding=padding_option,dilation_rate = [1,acc_rate]),convolutionBias); # convolution with dilation.                # activation
                    nonLinear_part_DeArtifact = tf.nn.leaky_relu(nonLinear_part_DeArtifact)
                else:
                    # regular CNN layer as output
                    nonLinear_part_DeArtifact = tf.nn.bias_add(tf.nn.convolution(nonLinear_part_DeArtifact, convolutionWeights,padding=padding_option,dilation_rate = [1,acc_rate]),convolutionBias); # convolution with dilation.                # activation
       
                    if (layer_idx != (layer_amount2-1)) :
                        nonLinear_part_DeArtifact = tf.nn.leaky_relu(nonLinear_part_DeArtifact)


            config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.42))
            sess = tf.Session(config=config)  # tensorflow initialize
            init = tf.global_variables_initializer()
            sess.run(init)
           
            # filter the result    
            [linear_part_valid,nonLinear_part_Denoise_valid,nonLinear_part_DeArtifact_valid] = sess.run([linear_part_valid[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:],nonLinear_part_Denoise_valid[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:],nonLinear_part_DeArtifact[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:]])    
            '''
            nonLinear_part_Denoise_valid = sess.run(nonLinear_part_Denoise_valid[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:])
            linear_part_valid = linear_part_valid[:,x_start_nonlinear2:x_end_nonlinear2+1,y_start_nonlinear2:y_end_nonlinear2+1,:]
            '''
            x_start = x_start + x_start_nonlinear2
            x_end = x_end - x_start_nonlinear2
            y_start = y_start + y_start_nonlinear2
            y_end = y_end - y_start_nonlinear2
            # cut the linear parts into valid area accroding to non-linear part
   
           
            # put the estimations into k-space
            # Put results into k-space
   
           
            for targetIdx in range(1,acc_rate):
                y_start_t = y_start + targetIdx
                y_end_t = y_end + targetIdx
                linear_recon[0,x_start:x_end+1,y_start_t:y_end_t+1:acc_rate,ind_shift_Channel_pos] = linear_part_valid[:,:,::acc_rate,targetIdx-1]
                nonLinear_part_DeArtifact_recon[0,x_start:x_end+1,y_start_t:y_end_t+1:acc_rate,ind_shift_Channel_pos] = nonLinear_part_DeArtifact_valid[:,:,::acc_rate,targetIdx-1]
                nonLinear_part_Denoise_recon[0,x_start:x_end+1,y_start_t:y_end_t+1:acc_rate,ind_shift_Channel_pos] = nonLinear_part_Denoise_valid[:,:,::acc_rate,targetIdx-1]
           
            sess.close()
            tf.reset_default_graph()

    kspace_recon = nonLinear_part_Denoise_recon + linear_recon
   
    return [kspace_recon, nonLinear_part_DeArtifact_recon,nonLinear_part_Denoise_recon, linear_recon]



# this function just map a complex k-space into real by a factor of two.
# the output is real(input) and imag(input) concatinated along coil direction.
def ZcMapComplexToReal(inputComplexTensor):

    [dim_X, dim_Y, dim_Z] = np.shape(inputComplexTensor)
    output = np.zeros([dim_X, dim_Y, dim_Z*2],dtype=np.float32)
    output[:,:,0:dim_Z] = np.real(inputComplexTensor)
    output[:,:,dim_Z:dim_Z*2] = np.imag(inputComplexTensor)

    return output

def ZcMapComplexToReal_4d(inputComplexTensor):

    [dim_B,dim_X, dim_Y, dim_Z] = np.shape(inputComplexTensor)
    output = np.zeros([dim_B,dim_X, dim_Y, dim_Z*2],dtype=np.float32)
    output[:,:,:,0:dim_Z] = np.real(inputComplexTensor)
    output[:,:,:,dim_Z:dim_Z*2] = np.imag(inputComplexTensor)

    return output


# this function loads the
def ZcLoadDataAndDivideACS_real(inputRoute, scaling, phaseshiftflag):

    file = sio.loadmat(inputRoute)
    kspace = file['kspace2'] # get kspace

    kspace = np.squeeze(kspace[0,:,:,:])
    normalize = 1#scaling/np.max(abs(kspace[:]))
    kspace = np.multiply(kspace,normalize) # normalizing & scaling


    # remove the zeros
    [x_start, x_end, y_start, y_end] = ZcGetValidArea3D_actual2(kspace)
    kspace = kspace[x_start:x_end+1,y_start:y_end+1,:]
   
    [m1,n1,no_ch] = np.shape(kspace)# for no_inds = 1 here

    kspace_all = kspace;
    kx = np.transpose(np.int32([(range(1,m1+1))]))                          # notice here 1:m1 = 1:m1+1 in python
    ky = np.int32([(range(1,n1+1))])


    if (phaseshiftflag):# ==1 || phaseshiftflag == 'TRUE' || phaseshiftflag == True)
        phase_shifts = np.dot(np.exp(-1j * 2 * 3.1415926535 / m1 * (m1/2-1) * kx ),np.exp(-1j * 2 * 3.14159265358979 / n1 * (n1/2-1) * ky ))
        for channel in range(0,no_ch):
            kspace_all[:,:,channel] = np.multiply(kspace_all[:,:,channel],phase_shifts)


    mask = np.squeeze(np.sum(np.sum(np.abs(kspace),0),1))>0;  # here is a littble bit tricky, it was sum(sum(xx,1),3) in matlab, but numpy will erase the 1 dim automatically after sum.
    picks = np.where(mask == 1);    
   
    mask = np.squeeze(np.sum(np.sum(np.abs(kspace),0),1))>0;  # here is a littble bit tricky, it was sum(sum(xx,1),3) in matlab, but numpy will erase the 1 dim automatically after sum.
    picks = np.where(mask == 1);                                  # be aware here all picks in python = picks in matlab - 1
    d_picks = np.diff(picks,1)  # this part finds the ACS region. if no diff==1, means no continuous sample lines, then no_ACS_flag==1
    indic = np.where(d_picks == 1);

    mask_x = np.squeeze(np.sum(np.sum(np.abs(kspace),2),1))>0;
    picks_x = np.where(mask_x == 1);
    x_start = picks_x[0][0]
    x_end = picks_x[0][-1]

    if np.size(indic)==0:    # if there is no no continuous sample lines, it means no ACS in the input
        no_ACS_flag=1;       # set flag
        print('No ACS signal in input data, using individual ACS file.')
        matfn = 'data_diffusion/ACS_29.mat'    # read outside ACS in
        matfn = 'datasm_multiACS_july.mat'
        matfn = 'MB16_sli1and5_All_july_8ACS.mat'
        ACS = sio.loadmat(matfn)
        ACS = ACS['ACS']    
        #normalize = scaling/np.max(abs(ACS[:])) # Has to be the same scaling or it won't work
        ACS = np.multiply(ACS,normalize)
        ACS = ACS[5:10,:,:,:]
        [b2,m2,n2,no_ch2] = np.shape(ACS)# for no_inds = 1 here
        print('Totally ',b2,' frames for training data')
        kx2 = np.transpose(np.int32([(range(1,m2+1))]))                          # notice here 1:m1 = 1:m1+1 in python
        ky2 = np.int32([(range(1,n2+1))])
 
        if (phaseshiftflag):
            print('Phase shifts applied into individual ACS signal')
            phase_shifts = np.dot(np.exp(-1j * 2 * 3.1415926535 / m2 * (m2/2-1) * kx2 ),np.exp(-1j * 2 * 3.14159265358979 / n2 * (n2/2-1) * ky2 ))
            for channel in range(0,no_ch2):
                ACS[:,:,channel] = np.multiply(ACS[:,:,channel],phase_shifts)
               
        ACS_re = ZcMapComplexToReal_4d(ACS)
       
       
    else:
        no_ACS_flag=0;
        print('ACS signal found in the input data')
        indic = indic[1][:]
        center_start = picks[0][indic[0]];
        center_end = picks[0][indic[-1]+1];
        ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
        ACS_re = ZcMapComplexToReal(ACS)

    acc_rate = d_picks[0][0]
    kspace_re = ZcMapComplexToReal(kspace)

    '''
    [dim1,dim2,dim3] = np.shape(ACS_re)
    ACS4d = np.zeros([b2,dim1,dim2,dim3])
    ACS4d[0,:,:,:] = ACS_re
    '''
    ACS4d = ACS_re;
   
    [dim1,dim2,dim3] = np.shape(kspace_re)
    kspace4d = np.zeros([1,dim1,dim2,dim3])
    kspace4d[0,:,:,:] = kspace_re

    return [ACS4d, kspace4d, acc_rate, no_ACS_flag]


# this function puts the ACS into the reconstructed k-space.
# inputs are reconstruction result and k-space(includes ACS)
def ZcPutACSintoKspace(reconKsp, kspace):

    mask = np.squeeze(np.sum(np.sum(np.abs(kspace),0),1))>0;  # here is a littble bit tricky, it was sum(sum(xx,1),3) in matlab, but numpy will erase the 1 dim automatically after sum.
    picks = np.where(mask == 1);    
   
    mask = np.squeeze(np.sum(np.sum(np.abs(kspace),0),1))>0;  # here is a littble bit tricky, it was sum(sum(xx,1),3) in matlab, but numpy will erase the 1 dim automatically after sum.
    picks = np.where(mask == 1);                                  # be aware here all picks in python = picks in matlab - 1
    d_picks = np.diff(picks,1)  # this part finds the ACS region. if no diff==1, means no continuous sample lines, then no_ACS_flag==1
    indic = np.where(d_picks == 1);

    indic = indic[1][:]
    center_start = picks[0][indic[0]]
    center_end = picks[0][indic[-1]+1]


    final = np.copy(reconKsp)
    final[:,center_start:center_end+1,:] = kspace[:,center_start:center_end+1,:]
   
    return final


def ZcLoadKsp2(filename):
    scaling=1
    file = sio.loadmat(filename)
    kspace = file['kspace2'] # get kspace

    kspace = np.squeeze(kspace)
    normalize = 1#scaling/np.max(abs(kspace[:]))
    kspace = np.multiply(kspace,normalize) # normalizing & scaling


    # remove the zeros
    [x_start, x_end, y_start, y_end] = ZcGetValidArea3D_actual2(kspace)
    kspace = kspace[x_start:x_end+1,y_start:y_end+1,:]
   
    [m1,n1,no_ch] = np.shape(kspace)# for no_inds = 1 here

    kspace_re = ZcMapComplexToReal(kspace)
    [dim1,dim2,dim3] = np.shape(kspace_re)
    kspace4d = np.zeros([1,dim1,dim2,dim3])
    kspace4d[0,:,:,:] = kspace_re
   
    return kspace4d

###################################### Script Starts. ############################################

no_ACS_flag = 1;
dirname = 'MB16_sli5and9'
dataFile = 'MB16_sli1and5/ksp1.mat'
phaseshiftflag = 0
IterAmount_rRAKI = 5001
IterAmount_rRAKI2 = 601
trainingflag = 0

tf.reset_default_graph()
#ksp_y_start = 108
#ksp_y_end = 220

config = tf.ConfigProto()
# load data, now they are numpy array in real domain.
[ACS_re, kspace_re, acc_rate, no_ACS_flag] = ZcLoadDataAndDivideACS_real(dataFile, 1, phaseshiftflag)
ACS_re = np.float32(ACS_re)
kspace_re = np.float32(kspace_re)
ACS_re = ACS_re[:,:,:,:]
[dim1,dim2,dim3,no_channels] = np.shape(ACS_re)



######################################## rRAKI first ###########################################
# train weights, rRAKI
[rRAKI_weights_all_values, linear_weight_shape, weight_shape_list,weight_shape_list2, weight_shape_list3, layer_amount,layer_amount2,layer_amount3] = ZcTrain_rRAKI_Weights(np.float32(ACS_re),acc_rate,trainingflag)

# recon

rRAKI_weights_all_values=ZcLoadMassiveWeights('rrakiF3s59item2_2021mod2_all',no_channels,layer_amount,layer_amount2,layer_amount3)


######################################################## GRAPH START ##################################################
tf.reset_default_graph()
[dim1,dimX,dimY,dimZ] = np.shape(kspace_re)
no_channels = dimZ
padding_option = 'SAME'
inputKspace = tf.placeholder(tf.float32, np.shape(kspace_re))

# in SMS, the last line of MB data is not sampled, we have more R-1 lines. so for rescon stage remove the ending R-1lines.

[x_start_linear,x_end_linear,y_start_linear,y_end_linear,x_start_nonlinear,x_end_nonlinear,y_start_nonlinear,y_end_nonlinear]= ZcGetValidArea3D_analytical(inputKspace,linear_weight_shape,weight_shape_list[:,:],acc_rate)  
[x_start,x_end,y_start,y_end] = ZcGetValidArea3D_analytical_convolution_FinalResultSize(inputKspace,linear_weight_shape,weight_shape_list,acc_rate)

for ind_shift_Channel_pos in range(no_channels):
   
    # linear part
    #linear_weights = rRAKI_weights_all_values[(('linear_weights_channel%d') % ind_shift_Channel_pos)]
    #linear_part =  tf.nn.convolution(inputKspace, linear_weights, padding=padding_option, dilation_rate = [1,acc_rate])
    #linear_part_valid = sess.run(linear_part[:,x_start_linear:x_end_linear+1,y_start_linear:y_end_linear+1,:])      
    # non-linear part

    linear_part = tf.identity(inputKspace);    
    for layer_idx in range(layer_amount):      
        linear_part = ZcAddCNNlayer_Dilate_Conti(linear_part,tf.constant(rRAKI_weights_all_values[(('linear_weights_layer%d_channel%d')%((layer_idx+1),ind_shift_Channel_pos))]),acc_rate,padding_option)
   
    linear_part_valid = (linear_part[:,x_start_nonlinear:x_end_nonlinear+1,y_start_nonlinear:y_end_nonlinear+1,:])


    nonLinear_trans = tf.identity(inputKspace);    
    for layer_idx in range(layer_amount):
        if((layer_idx<layer_amount-1) and layer_idx!=0):
            # CNN layer with dense connection
            nonLinear_trans = tf.nn.leaky_relu(ZcAddCNNlayer_Dilate_Conti(nonLinear_trans,tf.constant(rRAKI_weights_all_values[(('nonlinear_weights_layer%d_F_trans_channel%d')%((layer_idx+1),ind_shift_Channel_pos))]),acc_rate,padding_option))
        else:
            # regular CNN layer as output
            nonLinear_trans = ZcAddCNNlayer_Dilate_Conti(nonLinear_trans,tf.constant(rRAKI_weights_all_values[(('nonlinear_weights_layer%d_F_trans_channel%d')%((layer_idx+1),ind_shift_Channel_pos))]),acc_rate,padding_option)

            if (layer_idx != (layer_amount-1)) :
                nonLinear_trans = tf.nn.leaky_relu(nonLinear_trans)    
    
    nonLinear_trans_valid = (nonLinear_trans[:,x_start_nonlinear:x_end_nonlinear+1,y_start_nonlinear:y_end_nonlinear+1,:])
               
    tf.add_to_collection("output", (linear_part_valid))#[:,x_start_nonlinear3:x_end_nonlinear3+1,y_start_nonlinear3:y_end_nonlinear3+1,:]))#,name = ('linear_part_valid_%d' % ind_shift_Channel_pos)))
    tf.add_to_collection("output", (nonLinear_trans_valid))#[:,x_start_nonlinear3:x_end_nonlinear3+1,y_start_nonlinear3:y_end_nonlinear3+1,:]))#,name = ('nonLinear_part_DeArtifact_valid_%d' % ind_shift_Channel_pos)))

OutputCollection = tf.get_collection('output')

######################################################## GRAPH END ##################################################

config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
sess = tf.Session(config=config)  # tensorflow initialize
init = tf.global_variables_initializer()
sess.run(init)  

'''
x_start = x_start + x_start_nonlinear2
x_end = x_end - x_start_nonlinear2
y_start = y_start + y_start_nonlinear2
y_end = y_end - y_start_nonlinear2
'''

#########################################################

for i in range(1,489+1):
    #tf.reset_default_graph()
    #sess.run(init)  
    time_Learn_start = time.time() # set timer  
    filename = dirname+(('/ksp%d') % (i)) + '.mat'
    kspace_re = ZcLoadKsp2(filename)
    kspace_re = np.float32(kspace_re)
    print('Reconstructing file: ', filename)
   
    

    linear_recon = (np.copy(kspace_re))    
    nonLinear_part_trans_recon = (np.zeros(np.shape(kspace_re)))

    #time_Learn_end = time.time();
    #print('Loading data #',i,' costs ',(time_Learn_end - time_Learn_start),' sec') # get time

    # filter the result    
   
    #time_Learn_start = time.time() # set timer

    # cut the linear parts into valid area accroding to non-linear part
    recon =  sess.run(OutputCollection,feed_dict={inputKspace:kspace_re}) 
   
    for ind_shift_Channel_pos in range(no_channels):
        # put the estimations into k-space
        # Put results into k-space
        idx_channel = ind_shift_Channel_pos*2
        #[linear_part_valid_r,nonLinear_part_Denoise_valid_r,nonLinear_part_DeArtifact_valid_r] = 
        linear_part_valid_r=recon[idx_channel]    
        nonLinear_part_trans_valid_r=recon[idx_channel+1]        
          
        for targetIdx in range(1,acc_rate):
            y_start_t = y_start + targetIdx
            y_end_t = y_end + targetIdx

            linear_recon[0,x_start:x_end+1,y_start_t:y_end_t+1:acc_rate,ind_shift_Channel_pos] = linear_part_valid_r[:,:,::acc_rate,targetIdx-1]
            nonLinear_part_trans_recon[0,x_start:x_end+1,y_start_t:y_end_t+1:acc_rate,ind_shift_Channel_pos] = nonLinear_part_trans_valid_r[:,:,::acc_rate,targetIdx-1]
           
    trans_recon = nonLinear_part_trans_recon
    
    time_Learn_end = time.time();
    print('Reconstruction of Frame #',i,' costs ',(time_Learn_end - time_Learn_start),' sec') # get time




    trans_recon = np.squeeze(trans_recon)
    grap_recon = np.squeeze(linear_recon)

   
    kspace_NEVER_TOUCH = (kspace_re[:,:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_re[:,:,:,np.int32(no_channels/2):no_channels],1j))
    kspace_NEVER_TOUCH = np.squeeze(kspace_NEVER_TOUCH)

    cbc_trans_recon_complex = np.copy(trans_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(trans_recon[:,:,np.int32(no_channels/2):no_channels],1j))

    cbc_grap_recon_complex = np.copy(grap_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(grap_recon[:,:,np.int32(no_channels/2):no_channels],1j))

    resname = dirname+(('/ksp%d_rRAKI_F3item2_2021mod2_all') % (i)) + '.mat'
    sio.savemat(resname,{'cbc_trans':cbc_trans_recon_complex,'cbc_g':cbc_grap_recon_complex})

    time_Learn_end = time.time();
    print('Reconstruction of Frame #',i,' costs ',(time_Learn_end - time_Learn_start),' sec including saving files') # get time
