# import deeplift 
import numpy as np 
# from deeplift.conversion import kerasapi_conversion as kc



def get_padded_data(data, mapping):
	padded_data = np.zeros(len(mapping))
	for orig_index in mapping.keys():
		new_index = mapping[orig_index]
		# print ('Mapping: Original ', orig_index, 'New index: ', new_index)
		if new_index != -1:
			padded_data[orig_index] = data[new_index]

	return padded_data

def get_masked_data(data, mask): #data shape is (num samples, num features)
	masked_data = []
	mapping = dict() #key is the original index, value is the new index
	new_index = 0
	for orig_index, value in enumerate(mask):
		if value == 1:
			mapping[orig_index] = new_index
			new_index += 1
		else:
			mapping[orig_index] = -1
	
	# print (mapping)
	for sample in data:
		masked_sample = np.zeros(int(np.sum(mask)))
		for orig_index in mapping.keys():
			new_index = mapping[orig_index]
			if new_index != -1:
				masked_sample[new_index] = sample[orig_index]

		masked_data.append(masked_sample)

	return np.array(masked_data), mapping

def remove_symmetry(matrix):
    new_matrix = []
    row = matrix.shape[0]
    col = matrix.shape[1]
    # row 1 take 0 to 115, row 2 take 1 to 115 and so on.
    for n in range(0, row): 
        new_matrix = np.append(new_matrix, matrix[n][n:col], axis=0)
    return new_matrix


