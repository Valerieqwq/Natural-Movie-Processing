# =============================================================
# 0(a). Initialize
# =============================================================

import warnings
import sys 
import os    

import deepdish as dd
import numpy as np

import brainiak.eventseg.event
import nibabel as nib
from nilearn.input_data import NiftiMasker

import scipy.io
from scipy import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve
from sklearn import decomposition
from sklearn.model_selection import LeaveOneOut, KFold

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 

sns.set(style = 'white', context='talk', font_scale=1, rc={"lines.linewidth": 2})

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

# data type 1

D = dd.io.load("sherlock.h5")
BOLD = D['BOLD']
coords = D['coords']
human_bounds = D['human_bounds']

number_region, number_TR, number_subject = BOLD.shape

# data type 2

# read subject data and labels
def read_subj_data_N_label(subj_id):
    
    def load_labels(subj_id): 
        label_fname  = 'labels_subj%d.mat' % subj_id
        labels       = scipy.io.loadmat('sherlock_dir/labels/' + label_fname)
        recall_times = labels['recall_scenetimes']
        movie_times  = labels['subj_movie_scenetimes']
        return recall_times, movie_times

    def read_sherlock_mat_data(fpath):
        temp = scipy.io.loadmat(fpath)
        return temp['rdata'].T
    
    # specify the general path name
    data_dir_subj = 'sherlock_dir/data_mat/s%d'%subj_id
    recall_dir    = '/sherlock_recall'
    movie_dir     = '/sherlock_movie'

    # load data
    fpath_aud_movie  = data_dir_subj + movie_dir  + '/aud_early_sherlock_movie_s%d.mat'%subj_id
    fpath_aud_recall = data_dir_subj + recall_dir + '/aud_early_sherlock_recall_s%d.mat'%subj_id

    fpath_pmc_movie  = data_dir_subj + movie_dir  + '/pmc_nn_sherlock_movie_s%d.mat'%subj_id
    fpath_pmc_recall = data_dir_subj + recall_dir + '/pmc_nn_sherlock_recall_s%d.mat'%subj_id

    data_aud_movie   = read_sherlock_mat_data(fpath_aud_movie)
    data_aud_recall  = read_sherlock_mat_data(fpath_aud_recall)

    data_pmc_movie   = read_sherlock_mat_data(fpath_pmc_movie)
    data_pmc_recall  = read_sherlock_mat_data(fpath_pmc_recall)

    # load labels
    recall_times, movie_times = load_labels(subj_id)
    
    # numbers of events
    assert recall_times.shape == movie_times.shape
    number_event = recall_times.shape[0]
    
    # gather them into a dict
    subj_data_N_label = {
        'data_aud_movie' : data_aud_movie,
        'data_aud_recall': data_aud_recall,
        'data_pmc_movie' : data_pmc_movie,
        'data_pmc_recall': data_pmc_recall,
        'recall_times'   : recall_times,
        'movie_times'    : movie_times,
        'number_event'   : number_event
        }
    
    return subj_data_N_label
    
def get_certain_subj(dict_key, subj_id=range(17)):
    """
    subj_id (iterator) : (!!! Start from zero !!!)
    """
    
    # load all the movie data
    all_data = {}
    for i in range(1, 18):
        subj_data_N_label = read_subj_data_N_label(i)
        data_movie = subj_data_N_label[dict_key]
        all_data[str(i)] = data_movie
    
    # create data matrix
    subj_data_set = []
    
    for i in subj_id:
        subj_data_set.append(all_data[str(i+1)])
    subj_data_set = np.mean(np.array(subj_data_set), axis=0)
    return subj_data_set
    
# ============================================================
# 0(b). Take a simple look at data
# ============================================================

# ------------------------------------------------------------
# See the timepoint-timepoint correlation
# (data type 1, subject average)
# ------------------------------------------------------------

# HMM_auditory = brainiak.eventseg.event.EventSegment(100)
# HMM_auditory.fit(get_certain_subj('data_aud_movie', range(17)))
#
# HMM_postmedial = brainiak.eventseg.event.EventSegment(50)
# HMM_postmedial.fit(get_certain_subj('data_pmc_movie', range(17)))
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
# title_text_1 = '''
# TR-TR correlation
# Early Auditory
# '''
# title_text_2 = '''
# TR-TR correlation
# Post Medial
# '''
# ax1.set_title(title_text_1)
# ax2.set_title(title_text_2)
# ax1.imshow(np.corrcoef(get_certain_subj('data_aud_movie')), cmap='viridis')
# ax2.imshow(np.corrcoef(get_certain_subj('data_pmc_movie')), cmap='viridis')
# ax1.set_xlabel('TR')
# ax1.set_ylabel('TR')
# ax2.set_xlabel('TR')
#
# # extract the boundaries
# bounds1 = np.where(np.diff(np.argmax(HMM_auditory.segments_, axis=1)))[0]
# # plot the boundaries
# bounds_aug1 = np.concatenate(([0],bounds1,[number_TR]))
# for i in range(len(bounds_aug1)-1):
#     rect = patches.Rectangle(
#         (bounds_aug1[i],bounds_aug1[i]),
#         bounds_aug1[i+1]-bounds_aug1[i],
#         bounds_aug1[i+1]-bounds_aug1[i],
#         linewidth=2,edgecolor='w',facecolor='none'
#     )
#     ax1.add_patch(rect)
# # extract the boundaries
# bounds2 = np.where(np.diff(np.argmax(HMM_postmedial.segments_, axis=1)))[0]
# bounds_aug2 = np.concatenate(([0],bounds2,[number_TR]))
# for i in range(len(bounds_aug2)-1):
#     rect = patches.Rectangle(
#         (bounds_aug2[i],bounds_aug2[i]),
#         bounds_aug2[i+1]-bounds_aug2[i],
#         bounds_aug2[i+1]-bounds_aug2[i],
#         linewidth=2,edgecolor='w',facecolor='none'
#     )
#     ax2.add_patch(rect)
#
# f.tight_layout()
# plt.show()

# ------------------------------------------------------------
# Find the events in this dataset
# ------------------------------------------------------------

# specify the number of events
# K = 50
#
# # create the model
# hmm_sim = brainiak.eventseg.event.EventSegment(K)
# # fit the model
# hmm_sim.fit(np.mean(BOLD, axis=2).T)
#
# f, ax = plt.subplots(1,1, figsize=(4, 4))
# ax.imshow(hmm_sim.event_pat_.T, cmap='viridis', aspect='auto')
# ax.set_title('Estimated brain pattern for each event ($m_k$)')
# ax.set_ylabel('Event id')
# ax.set_xlabel('Voxels')
#
# f, ax = plt.subplots(1,1, figsize=(4,4))
# pred_seg = hmm_sim.segments_[0]
# ax.imshow(pred_seg.T, aspect='auto', cmap='viridis')
# ax.set_xlabel('Timepoints')
# ax.set_ylabel('Event label')
# ax.set_title('Predicted event segmentation,\nby HMM with the ground truth n_events')
# f.tight_layout()
#
# plt.show()

# ------------------------------------------------------------
# find the best model
# ------------------------------------------------------------

# TODO: generalize to all subjects

def find_best_model(dict_key, K_range=range(10, 101, 10)):
    """
    dict_key (str): specify the data (early auditory or post medial)
    k_range (iterator): specify the range of K (e.g. range(10, 101, 10))
    
    return: best model (brainiak.eventseg.event.EventSegment)
    """
    
    print("\n" + '='*70 + "\n\tModel for " + dict_key + "\n" + '='*70 + "\n")
    
    # number of splits
    n_splits_inner = 4
    # all subject-id
    subj_id_all = np.array(list(range(number_subject)))

    # set up outer loop loo structure
    loo_outer = LeaveOneOut()
    loo_outer.get_n_splits(subj_id_all)
    
    def tune_K(K_range, train_subj_id, test_subj_id):
        # fit HMM with different K (e.g., 10, 20, ..., 100)
        list_ll = []
        for K in K_range:
            # fit the data to HMM
            HMM = brainiak.eventseg.event.EventSegment(K)
            HMM.fit(get_certain_subj(dict_key, train_subj_id))
            train_ll = HMM.ll_
            # get the log likelihood on some unseen subjects
            _, log_likelihood = HMM.find_events(
                                    get_certain_subj(dict_key, test_subj_id))
            # print the progress
            print("\t|| K =", K, ',', end='')
            print("\ttest log likelihood", log_likelihood)
            # append the log likelihood
            list_ll.append([K, log_likelihood])
        
        # # TODO: plot the log likelihood by different Ks
        # plt.plot(list_ll)
        # plt.title("Log Likelihood")
        # plt.show()
        # pick the best K
        K_best, optimal_ll = max(list_ll, key=lambda ls: ls[1])
        print("\tbest K:", K_best,
              "\n\tcertain log likelihood:", optimal_ll)
        
        return HMM, K_best
        
    # outer loop
    list_best_test = []
    for subj_id_train_outer, subj_id_test_outer in loo_outer.split(subj_id_all):
        
        print("-" * 70)
        print("Outer:\nTrain:", subj_id_train_outer,
              "Test:", subj_id_test_outer)
        print("-" * 70)

        # set up inner loop loo structure
        subj_id_all_inner = subj_id_all[subj_id_train_outer]
        kf = KFold(n_splits=n_splits_inner)
        kf.get_n_splits(subj_id_train_outer)

        # inner loop: tune K
        print('\n\t# Inner:\n')
        for subj_id_train_inner, subj_id_test_inner in kf.split(subj_id_all_inner):
            
            # inplace update the ids w.r.t. to the inner training set
            subj_id_train_inner = subj_id_all_inner[subj_id_train_inner]
            subj_id_test_inner = subj_id_all_inner[subj_id_test_inner]
            
            print("\tTrain:", subj_id_train_inner,
                  "\tTest:", subj_id_test_inner)
            
            inner_best_model, inner_best_K = tune_K(\
                K_range, subj_id_train_inner, subj_id_test_inner)
    
        # final test for outer loop
        _, outer_final_ll = inner_best_model.find_events(get_certain_subj\
                                     (dict_key, subj_id_test_outer))
        list_best_test.append((inner_best_K, outer_final_ll))
        print("\nTest for outer:")
        print("K:", inner_best_K, "test log likelihood:", outer_final_ll)
        
        
        # ####### 🚧🚧🚧🚧🚧🚧🚧
        # 
        # statistical testing of boundaries
        #
        # k = 60
        # w = 5  # window size
        # n_permutation = 1000
        #
        # within_across = np.zeros((nSubj, nPerm+1))
        # for left_out in range(nSubj):
        #     # Fit to all but one subject
        #     ev = brainiak.eventseg.event.EventSegment(k)
        #     ev.fit(BOLD[:,:,np.arange(nSubj) != left_out].mean(2).T)
        #     events = np.argmax(ev.segments_[0], axis=1)
        #
        #     # Compute correlations separated by w in time
        #     corrs = np.zeros(nTR-w)
        #     for t in range(nTR-w):
        #         corrs[t] = pearsonr(BOLD[:,t,left_out],BOLD[:,t+w,left_out])[0]
        #     _, event_lengths = np.unique(events, return_counts=True)
        #
        #     # Compute within vs across boundary correlations, for real and permuted bounds
        #     np.random.seed(0)
        #     for p in range(nPerm+1):
        #         within = corrs[events[:-w] == events[w:]].mean()
        #         across = corrs[events[:-w] != events[w:]].mean()
        #         within_across[left_out, p] = within - across
        #         #
        #         perm_lengths = np.random.permutation(event_lengths)
        #         events = np.zeros(nTR, dtype=np.int)
        #         events[np.cumsum(perm_lengths[:-1])] = 1
        #         events = np.cumsum(events)
        #     print('Subj ' + str(left_out+1) + ': within vs across = ' + str(within_across[left_out,0]))
        #
        #
        # ####### 🚧
        
    optimal_K, optimal_ll = max(list_best_test, key=lambda ls: ls[1])
    print("\tbest K:", optimal_K,
          "\n\tcertain log likelihood:", optimal_ll)
    
    HMM = brainiak.eventseg.event.EventSegment(K)
                
    return HMM
    
    
# very costly... (normally 17*4=68 fits and 68+17=85 evaluations)
HMM_auditory = find_best_model('data_aud_movie')
HMM_postmedial = find_best_model('data_pmc_movie')

# temporal: skip model-selection
HMM_auditory = brainiak.eventseg.event.EventSegment(100)
HMM_auditory.fit(get_certain_subj('data_aud_movie', range(17)))

HMM_postmedial = brainiak.eventseg.event.EventSegment(50)
HMM_postmedial.fit(get_certain_subj('data_pmc_movie', range(17)))


# ============================================================
# 1. Timescales of Cortical Event Segmentation
# ============================================================

"""
Use HMM (hidden Markov model) or the event segmentation model to 
separate the data into different events, it should quantitatively show 
that events segmented from low-level areas have fewer timepoints, and 
high-level areas have more timepoints. 
"""

# ------------------------------------------------------------
# Compare the time scale of events between regions
# ------------------------------------------------------------

# print the segments
auditory_segments = HMM_auditory.segments_[0]
postmedial_segments = HMM_postmedial.segments_[0]

print("auditory_segments\n", auditory_segments.shape)
ls_demo = []
for i in range(10):
    ls_demo.append([int(d) for d in list(auditory_segments[i,:])])
print(ls_demo)

# TODO: plot the event segments
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.imshow(auditory_segments.T, aspect='auto', cmap='viridis')
ax2.imshow(postmedial_segments.T, aspect='auto', cmap='viridis')
ax2.set_xlabel('Timepoints')
ax1.set_ylabel('Event label')
ax2.set_ylabel('Event label')
ax1.set_title('Auditory(Top) vs. Post Medial(Down)')

fig.tight_layout()
plt.show()

# TODO: quantitatively prove the difference

# ============================================================
# 2. Comparison of Event Boundaries
#    across Regions and to Human Annotations
# ============================================================

"""
Compare the events segmanted from the high-level areas with true events 
from the video
"""

# ------------------------------------------------------------
# get event boundaries
# ------------------------------------------------------------

# generated pattern

# human label
