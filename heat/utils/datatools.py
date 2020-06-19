from ..core import types
from ..core.communication import MPI_WORLD

import h5py
import numpy as np
import os

__all__ = ["merge_files_imagenet_tfrecord"]


def merge_files_imagenet_tfrecord(folder_name, output_name=None):
    """
    merge multiple files together, result is one HDF5 file with all of the images stacked in the 0th dimension

    Parameters
    ----------
    filenames : str, optional*
        names of the files to join, either filenames or folder_names must not be None
    folder_name : str, optional*
        folder location of the files to join, either filenames or folder_names must not be None
    save_dtype : types, optional
    flatten : bool, optional
    output_name : str, optional

    """
    """
    labels:
        image/encoded: string containing JPEG encoded image in RGB colorspace
        image/height: integer, image height in pixels
        image/width: integer, image width in pixels
        image/colorspace: string, specifying the colorspace, always 'RGB'
        image/channels: integer, specifying the number of channels, always 3
        image/format: string, specifying the format, always 'JPEG'
        image/filename: string containing the basename of the image file
                e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
        image/class/label: integer specifying the index in a classification layer.
                The label ranges from [1, 1000] where 0 is not used.
        image/class/synset: string specifying the unique ID of the label, e.g. 'n01440764'
        image/class/text: string specifying the human-readable version of the label
                e.g. 'red fox, Vulpes vulpes'
        image/object/bbox/xmin: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/xmax: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/ymin: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/ymax: list of integers specifying the 0+ human annotated bounding boxes
        image/object/bbox/label: integer specifying the index in a classification
                layer. The label ranges from [1, 1000] where 0 is not used. Note this is
                always identical to the image label."""

    # create output file
    if isinstance(output_name, str) and output_name[:-3] != ".h5":
        output_name += ".h5"
    elif not isinstance(output_name, str) and output_name is not None:
        raise TypeError("output_name must be a string")
    else:
        output_name = "imagenet_merged.h5"

    # todo: get the number of files, either from the filenames or the contents of the folder
    if folder_name is not None:
        train_names = [f for f in os.listdir(folder_name) if f.startswith("train")]
        val_names = [f for f in os.listdir(folder_name) if f.startswith("val")]
    num_train = len(train_names)
    num_val = len(val_names)
    # todo: split the files between the available processes
    rank = MPI_WORLD.rank

    def _find_output_name_and_stsp(num_names, output_name=output_name):
        if MPI_WORLD.size > 1:
            files_per_pr = num_names // MPI_WORLD.size
            rem = num_names % MPI_WORLD.size
            start = MPI_WORLD.rank * files_per_pr
            stop = (MPI_WORLD.rank + 1) * files_per_pr + 1
            if rank == MPI_WORLD.size - 1:
                # number of ranks is likely small so this is okay
                start += rem
            stop += rem
            # create multiple h5 files for each process
            output_name_lcl = str(rank) + output_name if rank > 0 else output_name
        else:
            start = 0
            stop = num_train + 1
            output_name_lcl = output_name
        return start, stop, output_name_lcl

    train_start, train_stop, output_name_lcl_train = _find_output_name_and_stsp(num_train)
    val_start, val_stop, output_name_lcl_val = _find_output_name_and_stsp(num_val)
    output_name_lcl_val = output_name_lcl_val[:-3] + "_val.h5"

    # create the output files
    train_lcl_file = h5py.File(output_name_lcl_train, "w")
    train_lcl_file.create_dataset(
        "images", (1282048 // MPI_WORLD.size, 179729), maxshape=(None, 179729), dtype="ui8"
    )
    train_lcl_file.create_dataset(
        "metadata", (1282048 // MPI_WORLD.size, 9), maxshape=(None, 9), dtype="f64"
    )
    train_lcl_file.create_dataset(
        "file_info", (1282048 // MPI_WORLD.size, 4), maxshape=(None, 4), dtype="s"
    )

    val_lcl_file = h5py.File(output_name_lcl_val, "w")
    val_lcl_file.create_dataset(
        "images", (50000 // MPI_WORLD.size, 179729), maxshape=(None, 179729), dtype="ui8"
    )
    val_lcl_file.create_dataset(
        "metadata", (50000 // MPI_WORLD.size, 9), maxshape=(None, 9), dtype="f64"
    )
    val_lcl_file.create_dataset(
        "file_info", (50000 // MPI_WORLD.size, 4), maxshape=(None, 4), dtype="s"
    )

    def __single_file_load(src):
        import tensorflow as tf

        # load a file and read it to a numpy array
        # src, entries = "train-01023-of-01024.tfrecord", 1252
        dataset = tf.data.TFRecordDataset(filenames=[src])
        imgs = []
        img_meta = [[] for _ in range(9)]
        file_arr = [[] for _ in range(4)]
        for c, _ in enumerate(dataset):
            raw_exmaple = next(iter(dataset))
            parsed = tf.train.Example.FromString(raw_exmaple.numpy())
            imgs.append(
                tf.io.decode_raw(
                    parsed.features.feature["image/encoded"].bytes_list.value[0], tf.uint8
                ).numpy()
            )
            img_meta[0].append(
                tf.cast(
                    parsed.features.feature["image/height"].int64_list.value[0], tf.float32
                ).numpy()
            )
            img_meta[1].append(
                tf.cast(
                    parsed.features.feature["image/width"].int64_list.value[0], tf.float32
                ).numpy()
            )
            img_meta[2].append(
                tf.cast(
                    parsed.features.feature["image/channels"].int64_list.value[0], tf.float32
                ).numpy()
            )
            img_meta[3].append(parsed.features.feature["image/class/label"].int64_list.value[0])
            img_meta[4].append(
                tf.cast(
                    parsed.features.feature["image/object/bbox/xmin"].float_list.value[0],
                    tf.float32,
                ).numpy()
            )
            img_meta[5].append(
                tf.cast(
                    parsed.features.feature["image/object/bbox/xmax"].float_list.value[0],
                    tf.float32,
                ).numpy()
            )
            img_meta[6].append(
                tf.cast(
                    parsed.features.feature["image/object/bbox/ymin"].float_list.value[0],
                    tf.float32,
                ).numpy()
            )
            img_meta[7].append(
                tf.cast(
                    parsed.features.feature["image/object/bbox/ymax"].float_list.value[0],
                    tf.float32,
                ).numpy()
            )
            img_meta[8].append(
                parsed.features.feature["image/object/bbox/label"].int64_list.value[0]
            )

            file_arr[0].append(parsed.features.feature["image/format"].bytes_list.value[0])
            file_arr[1].append(parsed.features.feature["image/filename"].bytes_list.value[0])
            file_arr[2].append(parsed.features.feature["image/class/synset"].bytes_list.value[0])
            file_arr[3].append(
                np.array(parsed.features.feature["image/class/text"].bytes_list.value[0])
            )

        out_images = np.array(imgs)
        # need to transpose because of the way that numpy understands nested lists
        out_img_meta = np.array(img_meta, dtype=np.float64).T
        out_file_array = np.array(file_arr, dtype=np.str).T
        return out_images, out_img_meta, out_file_array

    def __write_datasets(img_outl, img_metal, file_arrl, past_sizel, file):
        file["images"].resize((past_sizel + img_outl.shape[0], 179729))
        file["images"][past_sizel : img_outl.shape[0]] = img_outl
        file["metadata"].resize((past_sizel + img_metal.shape[0], 9))
        file["metadata"][past_sizel : img_metal.shape[0]] = img_metal
        file["file_info"].resize((past_sizel + img_metal.shape[0], 4))
        file["file_info"][past_sizel : img_metal.shape[0]] = file_arrl

    def __load_multiple_files(train_names, train_start, train_stop, file):
        loc_files = train_names[train_start:train_stop]
        img_out, img_meta, file_arr = None, None, None
        i, past_size = 0, 0
        for f in loc_files:  # train
            # this is where the data is created for
            imgs, img_metaf, file_arrf = __single_file_load(f)
            # create a larger ndarray with the results
            img_out = np.vstack(img_out, imgs) if img_out is not None else imgs
            img_meta = np.vstack(img_meta, img_metaf) if img_meta is not None else img_metaf
            file_arr = np.vstack(file_arr, file_arrf) if file_arr is not None else file_arrf
            # when 16 files are read then write to the lcl file
            i += 1
            if i % 16 == 15:
                __write_datasets(img_out, img_meta, file_arr, past_size, file)
                past_size = img_out.shape[0]
                img_out, img_meta, file_arr = None, None, None

        if img_out is not None:
            __write_datasets(img_out, img_meta, file_arr, past_size, file)

    __load_multiple_files(train_names, train_start, train_stop, train_lcl_file)
    __load_multiple_files(val_names, val_start, val_stop, val_lcl_file)

    if MPI_WORLD.size > 1 and rank == 0:
        # todo: merge the files together at the end

        for r in range(1, MPI_WORLD.size):
            rank_lcl_name = str(r) + output_name_lcl_train
            # get the size of the data, load half, write, load other half, write
            r_file = h5py.File(rank_lcl_name, "r")
            sz = r_file["images"].shape[0] // 2
            current_end = train_lcl_file["images"].shape[0]
            __write_datasets(
                r_file["images"][:sz],
                r_file["metadata"][:sz],
                r_file["file_info"][:sz],
                current_end,
                train_lcl_file,
            )
            current_end += sz
            __write_datasets(
                r_file["images"][sz:],
                r_file["metadata"][sz:],
                r_file["file_info"][sz:],
                current_end,
                train_lcl_file,
            )

            rank_lcl_name = str(r) + output_name_lcl_val
            # get the size of the data, load half, write, load other half, write
            r_file = h5py.File(rank_lcl_name, "r")
            sz = r_file["images"].shape[0] // 2
            current_end = train_lcl_file["images"].shape[0]
            __write_datasets(
                r_file["images"][:sz],
                r_file["metadata"][:sz],
                r_file["file_info"][:sz],
                current_end,
                val_lcl_file,
            )
            current_end += sz
            __write_datasets(
                r_file["images"][sz:],
                r_file["metadata"][sz:],
                r_file["file_info"][sz:],
                current_end,
                val_lcl_file,
            )

    #  add the column names to the datasets
    # labels:
    img_list = [1, 2, 4, 7, 10, 11, 12, 13, 14]
    file_list = [5, 6, 8, 9]
    feature_list = [
        "image/encoded",
        "image/height",
        "image/width",
        "image/colorspace",
        "image/channels",
        "image/format",
        "image/filename",
        "image/class/label",
        "image/class/synset",
        "image/class/text",
        "image/object/bbox/xmin",
        "image/object/bbox/xmax",
        "image/object/bbox/ymin",
        "image/object/bbox/ymax",
        "image/object/bbox/label",
    ]

    train_lcl_file["metadata"].attrs["column_names"] = [feature_list[l] for l in img_list]
    train_lcl_file["file_info"].attrs["column_names"] = [feature_list[l] for l in file_list]
    val_lcl_file["metadata"].attrs["column_names"] = [feature_list[l] for l in img_list]
    val_lcl_file["file_info"].attrs["column_names"] = [feature_list[l] for l in file_list]