import numpy as n
import pickle as pkl
import sys
import copy
import scipy.cluster.hierarchy as hac
import time
import matplotlib.pyplot as plt


def pkl_load(fname):
    load_file = open(fname, 'rb')
    loaded = pickle.load(load_file)
    load_file.close()
    return loaded

def pkl_save(fname, content):
    save_file = open(fname, 'wb')
    pickle.dump(content, save_file)
    save_file.close()
    print "pkl data stored in : ", fname


################## ################## ################## ################## ##################
##################     ##################     ##################     ##################
################## Define loading images for channels ##################
################## ################## ################## ################## ##################
##################     ##################     ##################     ##################

###### define area which is marked as found pixel
pl_area = 5

##### define number of slices and channels
no_channels = 4
no_slices = 13#19
##### define both thresholds
thr1 = 80#75#150. <- for C1 analysis#250. for C1:150 forC2:100  for C3:80  for C4:80
thr2 = 45#50##120. <- for C1 analysis  for C1:120 forC2:70  for C3:50  for C4:45

p_dist = 20

##### different thresholds for different channels
thr1_array = n.ones(no_slices) * thr1
thr2_array = n.ones(no_slices) * thr2

##### define radius of area for discarding noisy pixels
xlen = 4#4#4
ylen = xlen
sumfrac = 0.2#0.20 for C1:0.3 forC2:0.2  for C3:0.25  for C4:0.2
sumthr = 2. * xlen * 2. * ylen * sumfrac

sumfrac_array = n.ones(no_slices) * sumthr
radius_array = n.ones(no_slices) * xlen

xlong = 10
ylong = 10

slice_range = range(no_slices)

#######################################################################################
############## Complete analysis of one channel - all slices  #################
#######################################################################################

channel_string = 'C4' #'C1'

import scipy.ndimage as snd

total_neurons = []

for slice_idx in slice_range:
    ##### change loading scheme
    if (slice_idx+1) < 10:
        #slice_string ='./2315_plate2_3,2_r-only/C1-2315_plate2_3,2_r-only-000'+str(slice_idx+1)+'.txt'
        ####C1-2315_plate1_1,1_l-0001

        slice_string ='./2315_plate1_1,1_l/' + channel_string + '-2315_plate1_1,1_l-000'+str(slice_idx+1)


    if 100 > (slice_idx+1) > 9:
        #slice_string ='./2315_plate2_3,2_r-only/C1-2315_plate2_3,2_r-only-00'+str(slice_idx+1)+'.txt'
        slice_string ='./2315_plate1_1,1_l/' + channel_string + '-2315_plate1_1,1_l-00'+str(slice_idx+1)

    plate = n.loadtxt(slice_string)
    #plate=snd.gaussian_filter(plate, 5)

    ########### denoising first? ############


    thr = thr1_array[slice_idx]
    thr2 = thr2_array[slice_idx]

    ###### only consider pixel indices with intensity values larger than threshold
    plate_filt = plate > thr

    plate_nz = plate_filt.nonzero()

    nzx = copy.deepcopy(plate_nz[0])
    nzy = copy.deepcopy(plate_nz[1])

    #nzux = n.unique(nzx)
    #nzuy = n.unique(nzy)

    idx_length = range(len(nzx))
    start1 = time.time()

    print "start time measurement first loop/s"

    ################## introduce second threshold for sum of plate region.
    for idx in idx_length:
        xidx = nzx[idx]
        yidx = nzy[idx]
        proximity = (plate[xidx-xlen:xidx+xlen, yidx-ylen:yidx+ylen] > thr2).astype(int)
        if proximity.sum() < sumthr:
            nzx[idx] *= 0
            nzy[idx] *= 0


    stop1 = time.time()

    print "stopped time measurement second loop/s ", stop1 - start1

    ##### filter out pixels where less than sumthr percent of pixels were above thr2

    ##### remaining pixel indices
    nzx = nzx[nzx!=0]
    nzy = nzy[nzy!=0]

    plate_mod = copy.deepcopy(plate)

    if len(nzx)>1 and len(nzy)>1:

        ts = n.reshape([nzy,nzx], (2,len(nzx)))
        zpixels = hac.linkage(ts.T, method = 'average')

        ##### do not consider clustering if pixel values are above p_dist###
        #zpixels = zpixels_all[zpixels_all < p_dist]

        #zpix_prior = zpixels[::-1, 2]
        zpix_prior = zpixels[:, 2]

        zpixels_cut = zpixels[zpix_prior < p_dist]

        len_diffr = len(zpixels) - len(zpixels_cut)

        knee= n.diff(zpixels_cut[::-1, 2], 2)
        if len(knee) == 0:
            num_clust = 2
        else:
            ##### do not understand the '+2'
            num_clust = knee.argmax()+ len_diffr+1# +2# + len_diffr

        clusters = hac.fcluster(zpixels, num_clust, 'maxclust')

        nrn_list = n.zeros ( shape = (len(set(clusters)), 2))

        for clust, idx in zip(set(clusters), range(len(set(clusters))   )):
            coords = ts.T[clust == clusters]
            nrn_list[idx] = coords.mean(axis=0)

        nrn_list.astype(int)


        for cell_idx in range(len(nrn_list)):
            x_idx = nrn_list[cell_idx][1]
            y_idx = nrn_list[cell_idx][0]
            plate_mod[x_idx-pl_area:x_idx+pl_area, y_idx-pl_area:y_idx+pl_area] = 255.


    if len(nzx)==1 and len(nzy)==1:
        nrn_list = n.array([nzx[0], nzy[0]])
        nrn_list.astype(int)

        x_idx = nrn_list[1]
        y_idx = nrn_list[0]
        plate_mod[x_idx-pl_area:x_idx+pl_area, y_idx-pl_area:y_idx+pl_area] = 255.


    if len(nzx)==0 and len(nzy)==0:
        ##### no neurons found in slice
        nrn_list = n.array([])

    #plt.plot(range(1, len(zpixels)+1), zpixels[::-1, 2])

    plt.figure()
    plt.imshow(plate_mod)
    #plt.show()

    ##### saving figure #####
    if (slice_idx+1) < 10:
   	#mod_slice_string ='./2315_plate2_3,2_r-only/marked_' + channel_string +'-2315_plate2_3,2_r-only-000'+str(slice_idx+1)+'.png'
        mod_slice_string ='./'+ channel_string+'_2315_plate1_1,1_l_filtered/marked_' + channel_string + '-2315_plate1_1,1_l-000'+str(slice_idx+1)+'.png'
    if 100 > (slice_idx+1) > 9:
        #mod_slice_string ='./2315_plate2_3,2_r-only/marked_'+channel_string  + '-2315_plate2_3,2_r-only-00'+str(slice_idx+1)+'.png'
        mod_slice_string ='./'+channel_string+'_2315_plate1_1,1_l_filtered/marked_' + channel_string + '-2315_plate1_1,1_l-00'+str(slice_idx+1)+'.png'

   # plt.savefig(mod_slice_string)

    total_neurons.append(nrn_list)

    #if slice_idx == 11:
    #    sys.exit()

    #### here apply clustering of indices of plxy - then choose clustered indices for one plane
    #### afterwards look for best match in clustering


##################### here count neurons correctly in in-slice elements #####################

y_overlap = 20
x_overlap = y_overlap

nonzero_idc = []
##### first neuron with nonzero neurons stored in it
for slice_idx in slice_range:
    if len(total_neurons[slice_idx]) !=0:
        nonzero_idc.append( slice_idx)

nonzero_first = nonzero_idc[0]

if total_neurons[nonzero_first].shape == (2,):
    stored_neurons = copy.deepcopy(total_neurons[nonzero_first])
else:
    stored_neurons = copy.deepcopy(total_neurons[nonzero_first][0])

for slice_idx in nonzero_idc:
    print "slice_idx : ", slice_idx
    if len(total_neurons[slice_idx].shape) ==1:

        nrn_coords = total_neurons[slice_idx]
        if len(nrn_coords) !=0:
            stored_neurons = n.vstack((stored_neurons, nrn_coords))

    else:
        for nrn_idx in xrange(len(total_neurons[slice_idx])):
            nrn_coords = total_neurons[slice_idx][nrn_idx]
            stored_neurons = n.vstack((stored_neurons, nrn_coords))
    #if slice_idx ==1:
    #    sys.exit()


num_final = copy.deepcopy(stored_neurons[:2])

stored_neurons = stored_neurons.tolist()


while len(stored_neurons) >0:

    print "len_num_final: ", len(num_final)

    y_accept = True
    x_accept = True

    for fin_idx in range(len(num_final)):
        x_accept = n.abs(num_final[fin_idx][0] - stored_neurons[-1][0]) > x_overlap
        y_accept = n.abs(num_final[fin_idx][1] - stored_neurons[-1][1]) > y_overlap

        if (not y_accept) and (not x_accept):
            stored_neurons.pop()
            break

        if len(stored_neurons) ==0:
            break

    if x_accept or y_accept:
        num_final = n.vstack((num_final, n.array([stored_neurons[-1][0], stored_neurons[-1][1]])))

num_final =num_final[1:]

plate_marked = copy.deepcopy(plate)

for cell_idx in range(len(num_final)):
    x_idx = num_final[cell_idx][1]
    y_idx = num_final[cell_idx][0]
    plate_marked[x_idx-pl_area:x_idx+pl_area, y_idx-pl_area:y_idx+pl_area] = 255.

plt.figure()
plt.imshow(plate_marked)

###zps = hac.linkage(num_final, method = 'average')
###knee= n.diff(zps[::-1, 2], 2)
###num_clust = knee.argmax() +2
###clusters = hac.fcluster(zps, num_clust, 'maxclust')
###nrn_filt = n.zeros ( shape = (len(set(clusters)), 2))
###for clust, idx in zip(set(clusters), range(len(set(clusters))   )):
###    coords = num_final[clust == clusters]
###    nrn_filt[idx] = coords.mean(axis=0)





#filtered_neurons = n.array(stored_neurons[0])
#
#filter_prox = 100.
#
#for nrn_i in range(len(stored_neurons)-1):
#    for nrn_j in range(nrn_i+1, len(stored_neurons)):
#        if n.linalg.norm(stored_neurons[nrn_i] - stored_neurons[nrn_j]) > filter_prox:
#            n.vstack((filtered_neurons, stored_neurons[nrn_i]))





sys.exit()

###total_nfilt = []
###
###for slice_idx in nonzero_idc:
###    ##### check for overlapping entries in one slice
###    nrn_slice = total_neurons[slice_idx]
###    if len(nrn_slice) ==1:
###        stored_sln = copy.deepcopy(nrn_slice)
###    else:
###        stored_sln = copy.deepcopy(total_neurons[nonzero_first][0])
###
###    next_x = total_neurons[slice_idx].T[0]
###    next_y = total_neurons[slice_idx].T[1]
###
###    for detect_idx in range(len(nrn_slice))[:-1]:
###        xy_idc = total_neurons[slice_idx][detect_idx]
###
###        stored_x = stored_sln[stored_nrn_idx].T[0]
###        stored_y = stored_sln[stored_nrn_idx].T[1]
###
###        xy_accept = (n.abs(next_x - stored_x) > x_overlap) or (n.abs(next_y - stored_y) > y_overlap)
###
###
###sys.exit()
###
###stored_neurons = copy.deepcopy(total_neurons[nonzero_first])
###
####if len(stored_neurons) !=1:
###    ####### perform overlap analysis here again:
###
###
###
###for slice_idx in nonzero_idc[1:-1]:
###
###    next_x = total_neurons[slice_idx].T[0]
###    next_y = total_neurons[slice_idx].T[1]
###    ##### iterate over all indices in already stored neuron array
###    for stored_nrn_idx in range(len(stored_neurons)):
###        stored_x = stored_neurons[stored_nrn_idx].T[0]
###        stored_y = stored_neurons[stored_nrn_idx].T[1]
###        ##### compare with all elements in array of next_xy values
###
###        #if (n.abs(current_x-next_x) < x_overlap) and (n.abs(current_y-next_y) < y_overlap):
###        xy_accept = (n.abs(next_x - stored_x) > x_overlap) or (n.abs(next_y - stored_y) > y_overlap)
###
###        new_neurons = total_neurons[slice_idx][xy_accept]
###
###        stored_neurons = n.vstack((stored_neurons, new_neurons))




    ##### diferences in intensitiy indices - zero means neighboring indices - > need to check for length of arrays inside area

####pixel_dist = 2
####
####nzx_diffr = nzx[1:]-nzx[:-1]
####nzy_diffr = nzy[1:]-nzy[:-1]
####
####nzx_dist = nzx_diffr.nonzero()[0]
####
####hmm = (nzx_diffr >pixel_dist).nonzero()[0]
####
####
####hmm = n.insert(hmm, 0, 0)
####hmm = n.insert(hmm, len(hmm), len(nzx_diffr))
####
####meanx_list=[]
####sys.exit()
####for idx in xrange(len(hmm[:-1])):
####    xidx = hmm[idx]
####    meanx_idx = int(nzx[xidx:idx+1].mean())
####    meanx_list.append(meanx_idx)
####
####
####sys.exit()

#
#nx_idx = nzx_diffr.nonzero()[0]
#ny_idx = nzy_diffr.nonzero()[0]
#
#range_indices = range(len(ndx_idx))
#
#valid_x = []
#valid_y = []
#
#for idx in range_indices:
#    valid_x.append(nzx[idx-1])
#
#minx = 10.
#miny = 10.
#
#
#
#for idx in nzx:
#    if nzy[idx] == nzy[idx-1]
#
#
#def give_pixel_indices(plate, int_thresh, filter_array):
#
#    ### bool array for intensities larger than given filter array
#    cell_indices = plate > int_thresh
#
#    return cell_indices
