import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def PlotHistMjj(m_jj,title):

    plt.hist(m_jj,bins=50)
    plt.title(title)
    plt.savefig(title+'_mjj.png')
    plt.close()

    return


def read_h5_file(file_path):
    """Reads an HDF5 file and prints its contents."""
    f=h5py.File(file_path, 'r')
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    #a_group_key = list(f.keys())[0]
        
    #print(f.get('jet1').keys())

    #print(f.get('jet_coords'))
    #print(np.array(f.get('jet1').get('4mom')))

    return f



def replace_zero_logpt_andthen_exp(jet):
    """Replace all values in jet[:, :, 0] that are 0 with 1. I don't remember why I did this. Maybe a rick to get log(1)=0?"""
    jet[:, :, 2] = np.where(jet[:, :, 2] == 0, -1000, jet[:, :, 2])
    jet[:, :, 2] = np.exp(jet[:, :, 2])
    return jet


def GetWhatIWant(f):

    ''' To get some of the features I'm interested in. For instance m_jj: The dijet mass. (Check the LHCO data description for more info) '''

    m_jj=f.get('jet_features')[:,6]

    jet_features=f.get('jet_features')[:]
    
    jet_1_feat=f.get('jet1').get('features')[:]
    jet_1_feat=replace_zero_logpt_andthen_exp(jet_1_feat)
    
    
    
    
    jet_1_4m=f.get('jet1').get('4mom')[:]
    
    jet_2_feat=f.get('jet2').get('features')[:]
    replace_zero_logpt_andthen_exp(jet_2_feat)
    #print(jet_2_feat[:, :, 2])

    jet_2_4m=f.get('jet2').get('4mom')[:]
     
     
    mask_jet_1=jet_1_feat[:, :, 2] != 0
    multi_jet1=np.expand_dims(np.sum(mask_jet_1, axis=1),axis=1)
    
    mask_jet_2=jet_2_feat[:, :, 2] !=0
    multi_jet2=np.expand_dims(np.sum(mask_jet_2, axis=1),axis=1)
    M_combined = np.stack([multi_jet1, multi_jet2], axis=1)
     
    print(M_combined)
    print(np.shape(M_combined))
    
    jet_coords=f.get('jet_coords')[:]
    #print(np.shape(jet_coords))
    jet_coords = np.concatenate([jet_coords, M_combined], axis=-1)
    

    
    return m_jj,jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords

def SelectSignalRegionSignal(m_jj,jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords):

    ''' Selecte the signal events in the signal region according to m_jj '''
    arr = np.array(m_jj)

    # Get indices of elements in the range [3300, 3700]
    selected_indices = np.where((arr >= 3300) & (arr <= 3700))[0]
    
    jet_features = jet_features[selected_indices,:]
    jet_1_feat = jet_1_feat[selected_indices,:,:]
    jet_2_feat = jet_2_feat[selected_indices,:,:]
    
    
    jet_1_4m = jet_1_4m[selected_indices,:,:]
    jet_2_4m = jet_2_4m[selected_indices,:,:]
    
    
    jet_coords = jet_coords[selected_indices,:,:]

    #print("Original indices:", selected_indices)
    #print("Selected elements:", jet_coords)
    #print(np.shape(jet_coords))
    return jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords
    

def SaveH5(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords,path):
    ''' Save data '''
    
    
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("jet_features", data=jet_features)
        h5f.create_dataset("jet_1_feat", data=jet_1_feat)
        h5f.create_dataset("jet_2_feat", data=jet_2_feat)
        
        h5f.create_dataset("jet_1_4m", data=jet_1_4m)
        h5f.create_dataset("jet_2_4m", data=jet_2_4m)
        
        
        h5f.create_dataset("jet_coords", data=jet_coords)



    return


def CreateTrainTestSignal(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords):
    
    ''' Separate test and train set for signal data '''
    
    
    #selected_indices = np.random.choice(int(len(jet_features)), size=int(len(jet_features)*.7), replace=False)
    selected_indices = np.random.choice(int(len(jet_features)), size=int(len(jet_features)-50000), replace=False)
    # Select elements using the indices
    jet_features_train = jet_features[selected_indices,:]
    jet_1_feat_train = jet_1_feat[selected_indices,:,:]
    jet_2_feat_train = jet_2_feat[selected_indices,:,:]
    
    jet_1_4m_train = jet_1_4m[selected_indices,:,:]
    jet_2_4m_train = jet_2_4m[selected_indices,:,:]
    
    
    jet_coords_train = jet_coords[selected_indices,:,:]
    
    
    
    jet_features_test = np.delete(jet_features, selected_indices,axis=0)
    jet_1_feat_test = np.delete(jet_1_feat, selected_indices,axis=0)
    jet_2_feat_test = np.delete(jet_2_feat, selected_indices,axis=0)
    
    jet_1_4m_test = np.delete(jet_1_4m, selected_indices,axis=0)
    jet_2_4m_test = np.delete(jet_2_4m, selected_indices,axis=0)
    
    
    jet_coords_test = np.delete(jet_coords, selected_indices,axis=0)
    
    '''
    print('hello')
    print(jet_coords_train)
    print(np.shape(jet_coords_train))

    print(jet_coords_test)
    print(np.shape(jet_coords_test))
    '''
    return jet_features_train,jet_1_feat_train,jet_2_feat_train, jet_1_4m_train,jet_2_4m_train     , jet_coords_train, jet_features_test,jet_1_feat_test,jet_2_feat_test, jet_1_4m_test,jet_2_4m_test,jet_coords_test


def Create200kBKG(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords):

    ''' Obtain the pure bkg training data  (in the signal region) '''

    selected_indices = np.random.choice(int(len(jet_features)), size=200000, replace=False)


    bg_jet_features_train = jet_features[selected_indices,:]
    bg_jet_1_feat_train = jet_1_feat[selected_indices,:,:]
    bg_jet_2_feat_train = jet_2_feat[selected_indices,:,:]
    
    bg_jet_1_4m_train = jet_1_4m[selected_indices,:,:]
    bg_jet_2_4m_train = jet_2_4m[selected_indices,:,:]
    
    bg_jet_coords_train = jet_coords[selected_indices,:,:]



    bg_jet_features_rest = np.delete(jet_features, selected_indices,axis=0)
    bg_jet_1_feat_rest = np.delete(jet_1_feat, selected_indices,axis=0)
    bg_jet_2_feat_rest = np.delete(jet_2_feat, selected_indices,axis=0)
    
    
    
    bg_jet_1_4m_rest = np.delete(jet_1_4m, selected_indices,axis=0)
    bg_jet_2_4m_rest = np.delete(jet_2_4m, selected_indices,axis=0)
    
    bg_jet_coords_rest = np.delete(jet_coords, selected_indices,axis=0)



    return bg_jet_features_train,bg_jet_1_feat_train,bg_jet_2_feat_train,bg_jet_1_4m_train,bg_jet_2_4m_train,bg_jet_coords_train, bg_jet_features_rest,bg_jet_1_feat_rest,bg_jet_2_feat_rest,bg_jet_1_4m_rest,bg_jet_2_4m_rest,bg_jet_coords_rest


def Create100kBKG(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords):
    
    '''Obtain the training BKG  data to which signal data will be injected  (in the signal region) '''

    selected_indices = np.random.choice(int(len(jet_features)), size=100000, replace=False)
    
    bg_jet_features_train_dat = jet_features[selected_indices,:]
    bg_jet_1_feat_train_dat = jet_1_feat[selected_indices,:,:]
    bg_jet_2_feat_train_dat = jet_2_feat[selected_indices,:,:]
    
    bg_jet_1_4m_train_dat = jet_1_4m[selected_indices,:,:]
    bg_jet_2_4m_train_dat = jet_2_4m[selected_indices,:,:]
    
    bg_jet_coords_train_dat = jet_coords[selected_indices,:,:]



    bg_jet_features_rest = np.delete(jet_features, selected_indices,axis=0)
    bg_jet_1_feat_rest = np.delete(jet_1_feat, selected_indices,axis=0)
    bg_jet_2_feat_rest = np.delete(jet_2_feat, selected_indices,axis=0)
    
    
    
    bg_jet_1_4m_rest = np.delete(jet_1_4m, selected_indices,axis=0)
    bg_jet_2_4m_rest = np.delete(jet_2_4m, selected_indices,axis=0)
    
    bg_jet_coords_rest = np.delete(jet_coords, selected_indices,axis=0)

    return bg_jet_features_train_dat,bg_jet_1_feat_train_dat,bg_jet_2_feat_train_dat,bg_jet_1_4m_train_dat,bg_jet_2_4m_train_dat,bg_jet_coords_train_dat, bg_jet_features_rest,bg_jet_1_feat_rest,bg_jet_2_feat_rest,bg_jet_1_4m_rest,bg_jet_2_4m_rest,bg_jet_coords_rest



def Create50kTestBKG(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords):

    ''' Create bkg data test set (50K events)   (in the signal region) '''
    selected_indices = np.random.choice(int(len(jet_features)), size=150000, replace=False)
    
    bg_jet_features_test = jet_features[selected_indices,:]
    bg_jet_1_feat_test = jet_1_feat[selected_indices,:,:]
    bg_jet_2_feat_test = jet_2_feat[selected_indices,:,:]
    
    bg_jet_1_4m_test = jet_1_4m[selected_indices,:,:]
    bg_jet_2_4m_test = jet_2_4m[selected_indices,:,:]
    
    bg_jet_coords_test = jet_coords[selected_indices,:,:]



    return bg_jet_features_test,bg_jet_1_feat_test,bg_jet_2_feat_test,bg_jet_1_4m_test,bg_jet_2_4m_test,bg_jet_coords_test


def CreateInjectedTrainSamples(bg_jet_features_train,bg_jet_1_feat_train,bg_jet_2_feat_train, bg_jet_1_4m_train,bg_jet_2_4m_train   ,bg_jet_coords_train,jet_features_train,jet_1_feat_train,jet_2_feat_train,jet_1_4m_train,jet_2_4m_train,jet_coords_train,save_path):

    ''' Create bkg (the 100k) + injected signals (in the signal region)  '''
    list_of_signal_sizes=[10000,5000,2000,1000,600,300]

    for signal_size in list_of_signal_sizes:

        selected_indices = np.random.choice(len(jet_features_train), size=signal_size, replace=False)  # Unique indices
        
        jet_features_train_sel = jet_features_train[selected_indices,:]
        jet_1_feat_train_sel = jet_1_feat_train[selected_indices,:,:]
        jet_2_feat_train_sel = jet_2_feat_train[selected_indices,:,:]
        
        
        jet_1_4m_train_sel = jet_1_4m_train[selected_indices,:,:]
        jet_2_4m_train_sel = jet_2_4m_train[selected_indices,:,:]
        
        jet_coords_train_sel = jet_coords_train[selected_indices,:,:]
        
        
        jet_features_train_sel = np.concatenate((bg_jet_features_train, jet_features_train_sel), axis=0)
        
        #print(jet_features_train_sel)
        np.random.shuffle(jet_features_train_sel)
        #print(jet_features_train_sel)
     
        

        jet_1_feat_train_sel = np.concatenate((bg_jet_1_feat_train, jet_1_feat_train_sel), axis=0)
        np.random.shuffle(jet_1_feat_train_sel)
        
        jet_2_feat_train_sel = np.concatenate((bg_jet_2_feat_train, jet_2_feat_train_sel), axis=0)
        np.random.shuffle(jet_2_feat_train_sel)
        
        
        jet_1_4m_sel = np.concatenate((bg_jet_1_4m_train, jet_1_4m_train_sel), axis=0)
        np.random.shuffle(jet_1_feat_train_sel)
        
        jet_2_4m_train_sel = np.concatenate((bg_jet_2_4m_train, jet_2_4m_train_sel), axis=0)
        np.random.shuffle(jet_2_feat_train_sel)
        
        
        
        jet_coords_train_sel = np.concatenate((bg_jet_coords_train, jet_coords_train_sel), axis=0)
        np.random.shuffle(jet_coords_train_sel)
        
        
        #print(np.shape(jet_features_train_sel))
        path=save_path+'/Weak-mix-Train-'+str(signal_size)+'.h5'
        SaveH5(jet_features_train_sel,jet_1_feat_train_sel,jet_2_feat_train_sel,jet_1_4m_train_sel,jet_2_4m_train_sel,jet_coords_train_sel,path)
        
        

    return


######################################################

main_path='/net/data_ttk/hreyes/LHCO/processed_jg/original/'
save_path=main_path+'/ReasembleAL_1'

#os.makedirs(save_path,exist_ok=True)

#bg-N100-SR.h5    bg-N100.h5    sn-N100.h5
file_path=main_path+'bg_N100.h5'


f=read_h5_file(file_path)
print("read file")


m_jj=f.get('jet_features')[:,6]
print(np.shape(m_jj))

signal=np.sum(f.get('signal')[:])
#print(signal)
title='bg-sg'
PlotHistMjj(m_jj,title)

# m_jj,jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords=GetWhatIWant(f)



# jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords=SelectSignalRegionSignal(m_jj,jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords)


# jet_features_train,jet_1_feat_train,jet_2_feat_train,jet_1_4m_train,jet_2_4m_train,jet_coords_train, jet_features_test,jet_1_feat_test,jet_2_feat_test, jet_1_4m_test,jet_2_4m_test,jet_coords_test=CreateTrainTestSignal(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords)



# path=save_path+'/sn-N100-SR-Train.h5'
# SaveH5(jet_features_train,jet_1_feat_train,jet_2_feat_train,jet_1_4m_train,jet_2_4m_train,jet_coords_train,path)

# print(path)
# print(np.shape(jet_features_train))

# path=save_path+'/sn-N100-SR-Test.h5'
# SaveH5(jet_features_test,jet_1_feat_test,jet_2_feat_test,jet_1_4m_test,jet_2_4m_test,jet_coords_test,path)

# print(path)
# print(np.shape(jet_features_test))
# exit()
# ######################################


# file_path_bg=main_path+'/bg_N100_SR_extra.h5'


# f_bg=read_h5_file(file_path_bg)
# m_jj,jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords=GetWhatIWant(f_bg)




# bg_jet_features_train,bg_jet_1_feat_train,bg_jet_2_feat_train,bg_jet_1_4m_train,bg_jet_2_4m_train,bg_jet_coords_train, bg_jet_features_rest,bg_jet_1_feat_rest,bg_jet_2_feat_rest,bg_jet_1_4m_rest,bg_jet_2_4m_rest,bg_jet_coords_rest=Create200kBKG(jet_features,jet_1_feat,jet_2_feat,jet_1_4m,jet_2_4m,jet_coords)



# bg_jet_features_train_dat,bg_jet_1_feat_train_dat,bg_jet_2_feat_train_dat,bg_jet_1_4m_train_dat,bg_jet_2_4m_train_dat,bg_jet_coords_train_dat, bg_jet_features_rest,bg_jet_1_feat_rest,bg_jet_2_feat_rest,bg_jet_1_4m_rest,bg_jet_2_4m_rest,bg_jet_coords_rest=Create100kBKG(bg_jet_features_rest,bg_jet_1_feat_rest,bg_jet_2_feat_rest,bg_jet_1_4m_rest,bg_jet_2_4m_rest,bg_jet_coords_rest)


# bg_jet_features_test,bg_jet_1_feat_test,bg_jet_2_feat_test,bg_jet_1_4m_test,bg_jet_2_4m_test,bg_jet_coords_test=Create50kTestBKG(bg_jet_features_rest,bg_jet_1_feat_rest,bg_jet_2_feat_rest,bg_jet_1_4m_rest,bg_jet_2_4m_rest,bg_jet_coords_rest)





# path=save_path+'/bg-N100-SR-Train.h5'
# SaveH5(bg_jet_features_train,bg_jet_1_feat_train,bg_jet_2_feat_train, bg_jet_1_4m_train,bg_jet_2_4m_train,bg_jet_coords_train,path)

# print(path)
# print(np.shape(bg_jet_features_train))


# path=save_path+'/bg-N100-SR-Train-dat.h5'
# SaveH5(bg_jet_features_train_dat,bg_jet_1_feat_train_dat,bg_jet_2_feat_train_dat, bg_jet_1_4m_train_dat,bg_jet_2_4m_train_dat,bg_jet_coords_train_dat,path)

# print(path)
# print(np.shape(bg_jet_features_train_dat))



# path=save_path+'/bg-N100-SR-Test_150k.h5'
# SaveH5(bg_jet_features_test,bg_jet_1_feat_test,bg_jet_2_feat_test,bg_jet_1_4m_test,bg_jet_2_4m_test,bg_jet_coords_test,path)


# print(path)
# print(np.shape(bg_jet_features_test))

# ##############################################
# exit()
# CreateInjectedTrainSamples(bg_jet_features_train_dat,bg_jet_1_feat_train_dat,bg_jet_2_feat_train_dat, bg_jet_1_4m_train_dat,bg_jet_2_4m_train_dat  ,bg_jet_coords_train_dat,jet_features_train,jet_1_feat_train,jet_2_feat_train,jet_1_4m_train,jet_2_4m_train,jet_coords_train,save_path)




