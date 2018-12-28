from __future__ import print_function

import numpy as np
#import cmap.io.gct as gct
import pandas as pandas
from sklearn.metrics.pairwise import pairwise_distances
from cmapPy.pandasGEXpress.parse import parse
import os.path as osp

base_path = '.'
PICKLE_EXT = '.pickle'
NUMPY_EXT = '.npy' # if set differently, .npy will be appended anyway when saving
HDF5_EXT = '.h5'
MSGP_EXT = '.msg'
fname_in = 'bgedv2_QNORM.gctx'
fname_data_df = osp.join(base_path, 'bgedv2_QNORM_data_df' + HDF5_EXT)
fname_row_metadata_df = osp.join(base_path, 'bgedv2_QNORM_row_metadata_df' + HDF5_EXT)
fname_col_metadata_df = osp.join(base_path, 'bgedv2_QNORM_col_metadata_df' + HDF5_EXT)
fname_unique_inds = osp.join(base_path, 'bgedv2_QNORM_unique_inds' + NUMPY_EXT)
fname_data_df_uniq = osp.join(base_path, 'bgedv2_QNORM_uniq' + HDF5_EXT)

DIST_BATCH_SIZE = 4000
MIN_DIST_THRESHOLD = 1.0 # same as in D-GEX

## Raw data
if not osp.isfile(fname_data_df) or not osp.isfile(fname_row_metadata_df) or not osp.isfile(fname_col_metadata_df):
	print('Extracted DataFrames are not present. Extracting them from the GCTX file.')
	print('  Parsing.')
	pgex = parse(fname_in)
	print('  Parsed.')
	data_df = pgex.data_df.transpose()
	#print('  Saving.')
	#data_df.to_hdf(fname_data_df,'table')
	#pgex.row_metadata_df.to_hdf(fname_row_metadata_df,'table')
	#pgex.col_metadata_df.to_hdf(fname_col_metadata_df,'table')
	row_metadata_df = pgex.row_metadata_df
	col_metadata_df = pgex.col_metadata_df
	print('  Done.')

else:
	print('DataFrames extracted from the GCTX file are present. Loading them.')
	data_df = pd.read_pickle(fname_data_df)
	row_metadata_df = pd.read_pickle(fname_row_metadata_df)
	col_metadata_df = pd.read_pickle(fname_col_metadata_df)

n_samples, n_features = data_df.values.shape

print('Using data with n_samples={} and n_features={}.'.format(n_samples,n_features))
# Finding duplicates in data - i.e. if the euclidean distance of two samples is less than MIN_DIST_THRESHOLD, then they are considered to be duplicates
if not osp.isfile(fname_unique_inds):
	print('List of unique samples is not present. Founding them.')
	idx_i_list = []
	idx_j_list  = []
	for i in range(0,n_samples,DIST_BATCH_SIZE):
		for j in range(0,i+1,DIST_BATCH_SIZE):
			print('  Calculating distance for square starting at [{},{}].'.format(i,j))
			print(data_df.values[i:i+DIST_BATCH_SIZE, j:j+DIST_BATCH_SIZE].shape, len(data_df.values[i:i+DIST_BATCH_SIZE, j:j+DIST_BATCH_SIZE].flatten()) )
			if len(data_df.values[i:i+DIST_BATCH_SIZE, :]) == 0 or len(data_df.values[j:j+DIST_BATCH_SIZE, :]) == 0 :
				print('  Empty.')
				continue
			dsts = pairwise_distances(data_df.values[i:i+DIST_BATCH_SIZE,:],  data_df.values[j:j+DIST_BATCH_SIZE,:], n_jobs=1) 
			idx_i_sub, idx_j_sub = np.where(dsts < MIN_DIST_THRESHOLD)
			idx_i_list.extend(idx_i_sub)
			idx_j_list.extend(idx_j_sub)

	print('  Distances found, keeping only first occurence of duplicate items.')
	idx_to_remove = set()
	for i, j  in zip(idx_i_list, idx_j_list):
		if i != j:
			idx_to_remove.add(np.max((i,j)))

	idx_to_keep = set(range(n_samples)) - idx_to_remove
	unique_inds = np.array(sorted(idx_to_keep))
	np.save(fname_unique_inds,unique_inds)
	print('  Saved to {}.'.format(fname_unique_inds))
else:
	unique_inds = np.load(fname_unique_inds)

# Creating DataFrame containing only unique samples

data_df_uniq = data_df.iloc[unique_inds,:]
if not osp.isfile(fname_data_df_uniq):
	data_df_uniq.to_hdf(fname_data_df_uniq, 'table')