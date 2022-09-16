
#import print_function
import tensorflow as tf
import scipy.io as sio
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def weight_variable(shape,vari_name):                   # notice here, we can reuse those weight which gives fast convergence or the result from last iteration instead of using new random weight.
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])

#### LEANING FUNCTION ####
def learning(ACS_input,target_input,accrate_input,sess):
    # define placeholder for inputs to network
    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                   # 320*291*32 image size, 1 batch, ACS region here
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z])           # target size
    
    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         # only 1 sample, 320*291 in size, 32 channels (coils)

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)

    ker_conv = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, target_dim3],'G1')
    grp_conv = conv2d_dilate(Input, ker_conv ,accrate_input)

    x_shift = np.int32(np.floor(kernel_last_x/2))
    
    [aa,bb,dim_yy,cc]=np.shape(grp_conv);

    grap_y_start = np.int32(  (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * accrate_input  #-------------------------------------
    grap_y_end = np.int32(dim_yy) - np.int32(( (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * accrate_input - 1


    grap_y_start = np.int32(grap_y_start);
    grap_y_end = np.int32(grap_y_end+1);
    

    grapRes =  grp_conv[:,x_shift:x_shift+target_dim_X,grap_y_start:grap_y_end,:];
    # here 1 = floor(kernel_last_x/2)
    # and another 1 for y = floor(kernel_last_y/2)
    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1') 
    h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input)) 


    ## conv2 layer ##
    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2,accrate_input))

    ## conv3 layer ##
    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3,accrate_input)
    
    #print('shape grap result = ',np.shape(grp_conv))   
    #print('shape chosen grap = ',np.shape(residual))
    #print('shape h_conv3 = ',np.shape(h_conv3))
    # the error between prediction and real data

    #error_norm = tf.norm(input_Target - residual - h_conv3) + 1e-2*tf.norm(input_Target - residual)       # loss
    error_norm = 1*tf.norm(input_Target - grapRes - h_conv3) + 1*tf.norm(input_Target - grapRes) 
    train_step = tf.train.AdamOptimizer(1e-3).minimize(error_norm)
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(error_norm)	
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1  # initial error, we set 1 and it began to decrease.
    for i in range(2000):
        
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target})
        if i % 100 == 0:                                                                         # here if the improve of 50 iterations is small,
                                                                              # then we stop iteration. The threshold should 
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})     # be related to the kspace values, i.e. if the 

            print('The',i,'th iteration gives an error',error_now)                              # change is smaller than 1%, we quit etc.
            '''
            if abs(error_prev - error_now) < 1e-4:       # here the threshold is just a small number, this part can be improved with more reasonable choice.
                break
            else:
                error_prev = error_now
            '''
        
    error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    return [sess.run(ker_conv),sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),error]   #-------------------------------------

#### RECON CONVOLUTION FUNCTION ####
def cnn_3layer(input_kspace,gker,w1,b1,w2,b2,w3,b3,acc_rate,sess):                 #-------------------------------------

    
    grap = conv2d_dilate(input_kspace, gker ,acc_rate)
    x_shift = np.int32(np.floor(kernel_last_x/2))
    
    [aa,dim_x,dim_yy,dd] = np.shape(grap);

    grap_y_start = np.int32((np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate  #-------------------------------------
    grap_y_end = np.int32(dim_yy) - np.int32(( (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate - 1

    grap_y_start = np.int32(grap_y_start);
    grap_y_end = np.int32(grap_y_end+1);

    effectiveGrappa =  grap[:, x_shift:dim_x-x_shift, grap_y_start:grap_y_end, :];

    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate) 
    print('grap res shape = ',np.shape(grap))
    print('effective grap shape = ',np.shape(effectiveGrappa))
    print('h_conv shape = ',np.shape(h_conv3))
    return sess.run(effectiveGrappa+h_conv3), sess.run(effectiveGrappa),sess.run(h_conv3), sess.run(grap) # run the convolution                                        #-------------------------------------


def ZcShowRSSQ(img):
    scale = np.max(abs(img[:]))
    img = img/scale
    plt.figure()
    plt.imshow(img,cmap = 'gray',vmax=0.8)

def ZcKsp2Img(inputksp):
    recon= np.copy(inputksp)
    for sli in range(0,np.int32(16)):
        recon[:,:,sli] = np.fft.ifft2(recon[:,:,sli])
        
    recon = np.sum(np.abs(recon),2)
    ZcShowRSSQ(recon)

###############################  MAIN FUNCTION START ################################

####################### ----- INITIALIZE BLAH BLAH ----- ############################



#######################################################################
###                                                                 ###
### For convinience, everything are the same with Matlab version :) ###
###                                                                 ###
#######################################################################

no_ACS_flag = 0;

resultName = 'ResNet'
matfn = 'fastMRI_Brain.mat'
rate = 2;
ACSrange = 48
phaseshiftflag = 0;
kspace = sio.loadmat(matfn)
kspace = kspace['kspace'] # get kspace
#kspace = kspace[:,:,0:10]
mask = np.zeros_like(kspace)
[row,col,coil] = mask.shape
mask[:,::rate,:] = 1;
midpoi = col//2;
mask[:,midpoi-ACSrange//2+1:midpoi+ACSrange//2,:]=1
kspace = kspace*mask
# scale to 0.015 as usual
normalize = 1/np.max(abs(kspace[:]))
kspace = np.multiply(kspace,normalize)   
'''
scaling = 1e3
kspace = np.multiply(kspace,scaling)   
'''
[m1,n1,no_ch] = np.shape(kspace)# for no_inds = 1 here
no_inds = 1

kspace_all = kspace;
kx = np.transpose(np.int32([(range(1,m1+1))]))                          # notice here 1:m1 = 1:m1+1 in python
ky = np.int32([(range(1,n1+1))])

if phaseshiftflag ==1:
    phase_shifts = np.dot(np.exp(-1j * 2 * 3.1415926535 / m1 * (m1/2-1) * kx ),np.exp(-1j * 2 * 3.14159265358979 / n1 * (n1/2-1) * ky ))
    for channel in range(0,no_ch):
        kspace_all[:,:,channel] = np.multiply(kspace_all[:,:,channel],phase_shifts)


kspace = np.copy(kspace_all)
mask = np.squeeze(np.sum(np.sum(np.abs(kspace),axis=0),axis=1))>0;  # here is a littble bit tricky, it was sum(sum(xx,1),3) in matlab, but numpy will erase the 1 dim automatically after sum.
picks = np.where(mask == 1);                                  # be aware here all picks in python = picks in matlab - 1
kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  # this part erase the all zero columns before the 1st sampled column

kspace_NEVER_TOUCH = np.copy(kspace_all)

mask = np.squeeze(np.sum(np.sum(np.abs(kspace),axis=0),axis=1))>0;  # here is a littble bit tricky, it was sum(sum(xx,1),3) in matlab, but numpy will erase the 1 dim automatically after sum.
picks = np.where(mask == 1);                                  # be aware here all picks in python = picks in matlab - 1
d_picks = np.diff(picks,1)  # this part finds the ACS region. if no diff==1, means no continuous sample lines, then no_ACS_flag==1
indic = np.where(d_picks == 1);

mask_x = np.squeeze(np.sum(np.sum(np.abs(kspace),axis=2),axis=1))>0;
picks_x = np.where(mask_x == 1);
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

if np.size(indic)==0:    # if there is no no continuous sample lines, it means no ACS in the input
    no_ACS_flag=1;       # set flag
    print('No ACS signal in input data, using individual ACS file.')
    matfn = 'data_diffusion/ACS_29.mat'    # read outside ACS in 
    matfn = 'RO6/ACS6.mat'
    ACS = sio.loadmat(matfn)
    ACS = ACS['ACS']     
    normalize = 0.015/np.max(abs(ACS[:])) # Has to be the same scaling or it won't work
    ACS = np.multiply(ACS,normalize*scaling)
    #ACS = ACS[249:400,:,:]

    [m2,n2,no_ch2] = np.shape(ACS)# for no_inds = 1 here
    no_inds = 1

    kx2 = np.transpose(np.int32([(range(1,m2+1))]))                          # notice here 1:m1 = 1:m1+1 in python
    ky2 = np.int32([(range(1,n2+1))])
    
    if phaseshiftflag ==1:
        phase_shifts = np.dot(np.exp(-1j * 2 * 3.1415926535 / m2 * (m2/2-1) * kx2 ),np.exp(-1j * 2 * 3.14159265358979 / n2 * (n2/2-1) * ky2 ))
        for channel in range(0,no_ch2):
            ACS[:,:,channel] = np.multiply(ACS[:,:,channel],phase_shifts)




    kspace = np.multiply(kspace,scaling)
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    print(ACS_dim_X)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
else:
    no_ACS_flag=0;
    print('ACS signal found in the input data')
    indic = indic[1][:]
    center_start = picks[0][indic[0]];
    center_end = picks[0][indic[-1]+1];

    print('%%%%%%%%%%%%%%%%%%%%%%%%%',center_start,center_end)

    ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    #ACS = ACS[100:220,:,:]
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)


acc_rate = d_picks[0][0]
no_channels = ACS_dim_Z*2


kernel_x_1 = 5
kernel_y_1 = 2

kernel_x_2 = 1
kernel_y_2 = 1

kernel_last_x = 3
kernel_last_y = 2

layer1_channels = 32
layer2_channels = 8

name_weight = resultName + ('_weight_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))
name_image = resultName + ('_image_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))


existFlag = os.path.isfile(name_image)

gker_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, acc_rate - 1, no_channels],dtype=np.float32)
w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels],dtype=np.float32)
w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)
w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,acc_rate - 1, no_channels],dtype=np.float32)    #-------------------------------------

b1_flag = 0;
b2_flag = 0;                         #-------------------------------------
b3_flag = 0;

if (b1_flag == 1):
    b1_all = np.zeros([1,1, layer1_channels,no_channels]);
else:
    b1 = []

if (b2_flag == 1):
    b2_all = np.zeros([1,1, layer2_channels,no_channels])
else:
    b2 = []

if (b3_flag == 1):
    b3_all = np.zeros([1,1, layer3_channels, no_channels])
else:
    b3 = []


target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1);  # remember every indicies need to -1 in python
target_x_end = np.int32(ACS_dim_X - target_x_start -1);  #-------------------------------------

################################ ----- initialize done, lets RAKI! ----- #######################################

time_ALL_start = time.time()

################################ ----- LEARNING PART! ----- #######################################

[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) # Batch, X, Y, Z
ACS = np.float32(ACS)  # here we use ACS instead of ACS_re for convinience


target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate;     #-------------------------------------
target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1;

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = acc_rate - 1

print('go!')
time_Learn_start = time.time() # set timer

errorSum = 0;
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/3; # avoid fully allocating.



for ind_c in range(ACS_dim_Z):
    sess = tf.Session(config=config)  # best serial performance is given by allocating full mem at once, and not release the resources after each iter

    # set target lines
    target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
    print('learning channel #',ind_c+1)
    time_channel_start = time.time()
    for ind_acc in range(acc_rate-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1  #-------------------------------------
        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
        target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

    # learning


    
    [gker,w1,w2,w3,error]=learning(ACS,target,acc_rate,sess)  # call the learning function   #-------------------------------------
    gker_all[:,:,:,:,ind_c]=gker;
    w1_all[:,:,:,:,ind_c] = w1
    w2_all[:,:,:,:,ind_c] = w2
    w3_all[:,:,:,:,ind_c] = w3                               #-------------------------------------
    time_channel_end = time.time()
    print('Time Cost:',time_channel_end-time_channel_start,'s')
    print('Norm of Error = ',error)
    errorSum = errorSum + error

    sess.close()
    tf.reset_default_graph()

time_Learn_end = time.time();
print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min') # get time

name_weight = 'rrakiweightRO6.mat';
#name_weight = 'ResNetWeight.mat'
sio.savemat(name_weight, {'gker':gker_all,'w1': w1_all,'w2': w2_all,'w3': w3_all})  # save the weights into directory.   #-------------------------------------


weightfile = sio.loadmat(name_weight)
gker_all = weightfile['gker'] # get kspace
w1_all = weightfile['w1'] # get kspace
w2_all = weightfile['w2'] # get kspace
w3_all = weightfile['w3'] # get kspace




################################ ----- RECON PART! ----- #######################################


#######################################################################
###                                                                 ###
### For convinience, everything are the same with Matlab version :) ###
###                                                                 ###
#######################################################################

kspace_recon_all = np.copy(kspace_all)
kspace_recon_all_nocenter = np.copy(kspace_all)

#for ind_c in range(1) #here size(kspace_all,4) =1, I just skip this loop.

kspace = np.copy(kspace_all)

over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
kspace_und = kspace
kspace_und[:,over_samp,:] = 0;
[dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
kspace_und_re = np.float32(kspace_und_re)
kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_recon = kspace_und_re

raki_recon = np.zeros_like(kspace_recon)
grap_recon = np.copy(kspace_recon)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/3 ; # avoid fully allocating.


for ind_c in range(0,no_channels):
    print('Reconstruting Channel #',ind_c+1)


    sess = tf.Session(config=config)  # tensorflow initialize
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)


    # grab w and b
    gker = np.float32(gker_all[:,:,:,:,ind_c])
    w1 = np.float32(w1_all[:,:,:,:,ind_c])
    w2 = np.float32(w2_all[:,:,:,:,ind_c])     #-------------------------------------
    w3 = np.float32(w3_all[:,:,:,:,ind_c])

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];                 #-------------------------------------
        
    [res,grap,raki,rawgrap] = cnn_3layer(kspace_und_re,gker,w1,b1,w2,b2,w3,b3,acc_rate,sess)  # call the convolution function           #-------------------------------------
    target_x_end_kspace = dim_kspaceUnd_X - target_x_start;
    
    for ind_acc in range(0,acc_rate-1):

        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1;                   #-------------------------------------
        target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc;
        kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = res[0,:,::acc_rate,ind_acc]

        grap_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = grap[0,:,::acc_rate,ind_acc]

        raki_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = raki[0,:,::acc_rate,ind_acc]



    sess.close()
    tf.reset_default_graph()

kspace_recon = np.squeeze(kspace_recon)

kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); # im_ind = 1, skip one dim


grap_recon = np.squeeze(grap_recon)

grap_recon_complex = (grap_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(grap_recon[:,:,np.int32(no_channels/2):no_channels],1j))

raki_recon = np.squeeze(raki_recon)

raki_recon_complex = (raki_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(raki_recon[:,:,np.int32(no_channels/2):no_channels],1j)) 


if no_ACS_flag == 0:  # if we have ACS region in kspace, put them back
    kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
#    grap_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
    print('ACS signal has been putted back')
else:
    print('No ACS signal is putted into k-space')



kspace_recon_all[:,:,:] = kspace_recon_complex; # im_ind = 1, skip one dim

for sli in range(0,no_ch):
    kspace_recon_all[:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,sli])
#    grap_recon_complex[:,:,sli] = np.fft.ifft2(grap_recon_complex[:,:,sli])
#    raki_recon_complex[:,:,sli] = np.fft.ifft2(raki_recon_complex[:,:,sli])



rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
sio.savemat('rraki.mat',{'kspace_all':kspace_recon_complex,'kspace_all_noACS':kspace_recon_all_nocenter,'grap_all':grap_recon_complex,'raki_all':raki_recon_complex,'rawgrap':rawgrap,'effectiveGrap':grap})  # save the results

time_ALL_end = time.time()
print('All process costs ',(time_ALL_end-time_ALL_start)/60,'mins')
print('Error Average in Training is ',errorSum/no_channels)



