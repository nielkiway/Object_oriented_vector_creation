import numpy as np
import h5py
import time
import pandas as pd


def getting_2D_data_from_h5_filtered_np_xy_switched_without_intensity_filter(h5_path, main_name, part_name, slice_name, show_info=False):

    # setting the start timer for time information
    start_time = time.time()

    # opening HDF5 file
    with h5py.File(h5_path, 'r') as h5:
        # check whether slice exists -> if not: empty array returned
        if slice_name in h5[main_name][part_name]:
            # X and Y Axis are changed to fit the OpenCV coordinate system
            X_Axis = np.array(h5[main_name][part_name][slice_name]['Y-Axis'][:]).astype(int)
            Area = np.array(h5[main_name][part_name][slice_name]['Area'][:]).astype(int)
            Intensity = np.array(h5[main_name][part_name][slice_name]['Intensity'][:]).astype(int)
            Y_Axis = np.array(h5[main_name][part_name][slice_name]['X-Axis'][:]).astype(int)

            # The following if condition is added because pre investigations showed that in some very rare cases
            # the dimensions of the arrays weren't exactly the same as there was an extra point on top of one of the
            # arrays. The dimension inequality led to instabilities.
            # If dimensions aren't equal the following code block is entered

            if not X_Axis.size == Y_Axis.size == Area.size == Intensity.size:

                # determine the lowest value among the different sizes
                size_arr = np.array([X_Axis.size, Y_Axis.size, Area.size, Intensity.size])
                min_size = size_arr.min()

                if X_Axis.size != min_size:
                    diff_size_x = X_Axis.size - min_size
                    X_Axis = np.delete(X_Axis, -diff_size_x)

                if Y_Axis.size != min_size:
                    diff_size_y = Y_Axis.size - min_size
                    Y_Axis = np.delete(Y_Axis, -diff_size_y)

                if Area.size != min_size:
                    diff_size_area = Area.size - min_size
                    Area = np.delete(Area, -diff_size_area)

                if Intensity.size != min_size:
                    diff_size_intensity = Intensity.size - min_size
                    Intensity = np.delete(Intensity, -diff_size_intensity)

            if show_info:
                print(str(X_Axis.size) + ' total data points found')

            # The following line of code stacks the single arrays to a table like structure.
            combos = np.stack((X_Axis, Y_Axis, Area, Intensity), axis=-1)

            # The following block filters out single outlier points by comparing the X and Y-values to min and max
            # values.
            median_x = np.median(combos[:, 0])
            median_y = np.median(combos[:, 1])
            std_x = int(combos[:, 0].std())
            std_y = int(combos[:, 1].std())
            low_limit_x = median_x - 2 * std_x
            low_limit_y = median_y - 2 * std_y
            high_limit_x = median_x + 2 * std_x
            high_limit_y = median_y + 2 * std_y

            combos = np.delete(combos, np.where(combos[:, 0] < low_limit_x), axis=0)
            combos = np.delete(combos, np.where(combos[:, 0] > high_limit_x), axis=0)
            combos = np.delete(combos, np.where(combos[:, 1] < low_limit_y), axis=0)
            combos = np.delete(combos, np.where(combos[:, 1] > high_limit_y), axis=0)

            # The following block filters out points where Area and Intensity values are equal to 0.
            area_zeros = np.where(combos[:, 2] == 0)
            intensity_zeros = np.where(combos[:, 3] == 0)
            zero_area_intensity_indices = np.intersect1d(area_zeros, intensity_zeros)
            combos_wo_only_zeros = np.delete(combos, zero_area_intensity_indices, axis=0)

            if show_info:
                print(str(combos_wo_only_zeros.shape[0]) + ' data points where area != 0 AND intensity != 0')

            # The following block is used for handling x,y-combinations occurring multiple times
            _, unique_indices = np.unique(combos_wo_only_zeros[:, [0, 1]], axis=0, return_index=True)
            combos_unique = combos_wo_only_zeros[unique_indices]

            if show_info:
                print(str(combos_unique.shape[0]) + ' unique x,y-combinations where area != 0 AND intensity != 0')

            # getting all the indices belonging to non unique x,y-combinations
            index_range = np.arange(combos_wo_only_zeros.shape[0])
            indices_of_interest = np.setdiff1d(index_range, unique_indices)

            combo_processed_array = np.empty([0, 4], dtype=int)
            return_array = np.copy(combos_wo_only_zeros)
            index_counter = 0
            indices_list = []

            if show_info:
                print("vor iterieren %s seconds ---" % (time.time() - start_time))

            # looping through all the indices belonging to non unique x,y-combinations
            for index in indices_of_interest:
                xy_combo = combos_wo_only_zeros[:, [0, 1]][index]
                # checking whether x,y combination has already been checked, if yes -> no action
                if np.where((combo_processed_array[:, 0] == xy_combo[0]) * (combo_processed_array[:, 1] == xy_combo[1]))[0].size == 0:
                    index_counter += 1
                    xy_combo = combos_wo_only_zeros[:, [0, 1]][index]

                    # getting all the indices of the currently checked x,y-combination
                    indices_relevant = np.where((combos_wo_only_zeros[:, 0] == xy_combo[0]) * (combos_wo_only_zeros[:, 1] == xy_combo[1]))[0]

                    # getting the max Area and Intensity of the currently checked x,y-combination
                    # this block would need to be replaced in case mean values are desired
                    max_area_of_combo = np.amax(combos_wo_only_zeros[:, 2][indices_relevant])
                    max_intensity_of_combo = np.amax(combos_wo_only_zeros[:, 3][indices_relevant])

                    # stacking x, y, maxArea and maxIntensity together
                    max_combos = np.stack((xy_combo[0], xy_combo[1], max_area_of_combo, max_intensity_of_combo), axis=-1)

                    # stacking the created combination on top of the copy of combos_wo_only_zeros
                    return_array = np.vstack((return_array, max_combos))

                    # adding the relevant indices to the indices_list and adding the created combination to combo_processed_array
                    indices_list.append(list(indices_relevant))
                    combo_processed_array = np.vstack((combo_processed_array, max_combos))

            # creating a an array with all the indices of multiple points and deleting those positions from combos_
            # wo_only_zeros_copy -> this way all the old x,y combinations occurring multiple times are replaced by the
            # generated combination with max Values
            indices_relevant = np.hstack(indices_list)
            return_array = np.delete(return_array, indices_relevant, axis=0)

        else:
            return_array = np.empty([0, 4], dtype=int)
            print('{} is not existing -> empty array created'.format(slice_name))

        if show_info:
            print("array creation took %s seconds ---" % (time.time() - start_time))

    return return_array


def get_max_intensity_whole_part(h5_path, part_name, max_slice_num):
    df = pd.DataFrame(columns=['Slice_num', 'maxInt', 'medianInt', 'meanInt', 'StdInt'])
    int_array = []

    # for every slice in the part the maximum, mean, median and standarddeviation of the intensity value distribution are calculated
    for num_slice in range(50, max_slice_num):  # 50 added because first 50 layers should be neglected
        slice_name = 'Slice' + str("{:05d}".format(num_slice + 1))
        array = getting_2D_data_from_h5_filtered_np_xy_switched_without_intensity_filter(h5_path, part_name, slice_name)
        maxInt = array[:, 3].max()
        medianInt = np.median(array[:, 3])
        meanInt = np.mean(array[:, 3])
        stdInt = np.std(array[:, 3])

        # here an arrays is created which contains all the intensity values of the whole part
        int_array = np.hstack((int_array, (array[:, 3]).astype(int)))

        df = df.append(
            {'Slice_num': "{:05d}".format(num_slice + 1), 'maxInt': maxInt, 'medianInt': medianInt, 'meanInt': meanInt,
             'StdInt': stdInt}, ignore_index=True)

    return int_array.astype(int), df, df['medianInt'].mean(), df['StdInt'].mean()


#######################################################################################################################


def getting_2D_data_from_h5_including_points_without_lasing(h5_path, main_name, part_name, slice_name, show_info=False):

    # setting the start timer for time information
    start_time = time.time()

    # opening HDF5 file
    with h5py.File(h5_path, 'r') as h5:
        # check whether slice exists -> if not: empty array returned
        if slice_name in h5[main_name][part_name]:
            # X and Y Axis are changed to fit the OpenCV coordinate system
            X_Axis = np.array(h5[main_name][part_name][slice_name]['Y-Axis'][:]).astype(int)
            Area = np.array(h5[main_name][part_name][slice_name]['Area'][:]).astype(int)
            Intensity = np.array(h5[main_name][part_name][slice_name]['Intensity'][:]).astype(int)
            Y_Axis = np.array(h5[main_name][part_name][slice_name]['X-Axis'][:]).astype(int)

            # The following if condition is added because pre investigations showed that in some very rare cases
            # the dimensions of the arrays weren't exactly the same as there was an extra point on top of one of the
            # arrays. The dimension inequality led to instabilities.
            # If dimensions aren't equal the following code block is entered

            if not X_Axis.size == Y_Axis.size == Area.size == Intensity.size:

                # determine the lowest value among the different sizes
                size_arr = np.array([X_Axis.size, Y_Axis.size, Area.size, Intensity.size])
                min_size = size_arr.min()

                if X_Axis.size != min_size:
                    diff_size_x = X_Axis.size - min_size
                    X_Axis = np.delete(X_Axis, -diff_size_x)

                if Y_Axis.size != min_size:
                    diff_size_y = Y_Axis.size - min_size
                    Y_Axis = np.delete(Y_Axis, -diff_size_y)

                if Area.size != min_size:
                    diff_size_area = Area.size - min_size
                    Area = np.delete(Area, -diff_size_area)

                if Intensity.size != min_size:
                    diff_size_intensity = Intensity.size - min_size
                    Intensity = np.delete(Intensity, -diff_size_intensity)

            if show_info:
                print(str(X_Axis.size) + ' total data points found')

            # The following line of code stacks the single arrays to a table like structure.
            combos = np.stack((X_Axis, Y_Axis, Area, Intensity), axis=-1)

            # The following block filters out single outlier points by comparing the X and Y-values to min and max
            # values.
            median_x = np.median(combos[:, 0])
            median_y = np.median(combos[:, 1])
            std_x = int(combos[:, 0].std())
            std_y = int(combos[:, 1].std())
            low_limit_x = median_x - 2 * std_x
            low_limit_y = median_y - 2 * std_y
            high_limit_x = median_x + 2 * std_x
            high_limit_y = median_y + 2 * std_y

            combos = np.delete(combos, np.where(combos[:, 0] < low_limit_x), axis=0)
            combos = np.delete(combos, np.where(combos[:, 0] > high_limit_x), axis=0)
            combos = np.delete(combos, np.where(combos[:, 1] < low_limit_y), axis=0)
            combos = np.delete(combos, np.where(combos[:, 1] > high_limit_y), axis=0)

            # The following block filters out points where Area and Intensity values are equal to 0.
            # area_zeros = np.where(combos[:, 2] == 0)
            # intensity_zeros = np.where(combos[:, 3] == 0)
            # zero_area_intensity_indices = np.intersect1d(area_zeros, intensity_zeros)
            # combos_wo_only_zeros = np.delete(combos, zero_area_intensity_indices, axis=0)

            #if show_info:
            #    print(str(combos_wo_only_zeros.shape[0]) + ' data points where area != 0 AND intensity != 0')

            combos_wo_only_zeros = combos # just because of laziness

            # The following block is used for handling x,y-combinations occurring multiple times
            _, unique_indices = np.unique(combos_wo_only_zeros[:, [0, 1]], axis=0, return_index=True)
            combos_unique = combos_wo_only_zeros[unique_indices]

            if show_info:
                print(str(combos_unique.shape[0]) + ' unique x,y-combinations where area != 0 AND intensity != 0')

            # getting all the indices belonging to non unique x,y-combinations
            index_range = np.arange(combos_wo_only_zeros.shape[0])
            indices_of_interest = np.setdiff1d(index_range, unique_indices)

            combo_processed_array = np.empty([0, 4], dtype=int)
            return_array = np.copy(combos_wo_only_zeros)
            index_counter = 0
            indices_list = []

            if show_info:
                print("vor iterieren %s seconds ---" % (time.time() - start_time))

            # looping through all the indices belonging to non unique x,y-combinations
            for index in indices_of_interest:
                xy_combo = combos_wo_only_zeros[:, [0, 1]][index]
                # checking whether x,y combination has already been checked, if yes -> no action
                if np.where((combo_processed_array[:, 0] == xy_combo[0]) * (combo_processed_array[:, 1] == xy_combo[1]))[0].size == 0:
                    index_counter += 1
                    xy_combo = combos_wo_only_zeros[:, [0, 1]][index]

                    # getting all the indices of the currently checked x,y-combination
                    indices_relevant = np.where((combos_wo_only_zeros[:, 0] == xy_combo[0]) * (combos_wo_only_zeros[:, 1] == xy_combo[1]))[0]

                    # getting the max Area and Intensity of the currently checked x,y-combination
                    # this block would need to be replaced in case mean values are desired
                    max_area_of_combo = np.amax(combos_wo_only_zeros[:, 2][indices_relevant])
                    max_intensity_of_combo = np.amax(combos_wo_only_zeros[:, 3][indices_relevant])

                    # stacking x, y, maxArea and maxIntensity together
                    max_combos = np.stack((xy_combo[0], xy_combo[1], max_area_of_combo, max_intensity_of_combo), axis=-1)

                    # stacking the created combination on top of the copy of combos_wo_only_zeros
                    return_array = np.vstack((return_array, max_combos))

                    # adding the relevant indices to the indices_list and adding the created combination to combo_processed_array
                    indices_list.append(list(indices_relevant))
                    combo_processed_array = np.vstack((combo_processed_array, max_combos))

            # creating a an array with all the indices of multiple points and deleting those positions from combos_
            # wo_only_zeros_copy -> this way all the old x,y combinations occurring multiple times are replaced by the
            # generated combination with max Values
            indices_relevant = np.hstack(indices_list)
            return_array = np.delete(return_array, indices_relevant, axis=0)

        else:
            return_array = np.empty([0, 4], dtype=int)
            print('{} is not existing -> empty array created'.format(slice_name))

        if show_info:
            print("array creation took %s seconds ---" % (time.time() - start_time))

    return return_array

#######################################################################################################################

def get_max_intensity_whole_part(h5_path, part_name, max_slice_num):
    df = pd.DataFrame(columns=['Slice_num', 'maxInt', 'medianInt', 'meanInt', 'StdInt'])
    int_array = []

    # for every slice in the part the maximum, mean, median and standarddeviation of the intensity value distribution are calculated
    for num_slice in range(50, max_slice_num):  # 50 added because first 50 layers should be neglected
        slice_name = 'Slice' + str("{:05d}".format(num_slice + 1))
        array = getting_2D_data_from_h5_filtered_np_xy_switched_without_intensity_filter(h5_path, part_name, slice_name)
        maxInt = array[:, 3].max()
        medianInt = np.median(array[:, 3])
        meanInt = np.mean(array[:, 3])
        stdInt = np.std(array[:, 3])

        # here an arrays is created which contains all the intensity values of the whole part
        int_array = np.hstack((int_array, (array[:, 3]).astype(int)))

        df = df.append(
            {'Slice_num': "{:05d}".format(num_slice + 1), 'maxInt': maxInt, 'medianInt': medianInt, 'meanInt': meanInt,
             'StdInt': stdInt}, ignore_index=True)

    return int_array.astype(int), df, df['medianInt'].mean(), df['StdInt'].mean()