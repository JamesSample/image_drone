import pandas as pd
from skimage import exposure
from skimage.util import img_as_ubyte
from sklearn import metrics

import nivapy3 as nivapy


def image_to_sample_df(area, equalise=False, dropna=True):
    """Convert image to dataframe.
    Args:
        area:     Int. Area of interest, from 1 to 6
        equalise: Bool. Whether to apply a 'linear stretch' to equalise the image histogram
                  for each band
        dropna:   Bool. Whether to remove pixel classified as 'NoData' or 'Other' from the
                  dataset
    Returns:
        Dataframe.
    """
    # Paths to images
    raw_path = f"/home/jovyan/shared/drones/frisk_oslofjord/harry_frisk_oslofjord_script/jes/raster/aligned/training/ne_akeroya_10cm_area_{area}.tif"
    man_path = f"/home/jovyan/shared/drones/frisk_oslofjord/harry_frisk_oslofjord_script/jes/raster/aligned/training/ne_akeroya_10cm_area_{area}_man_class.tif"

    # Container for data
    data_dict = {}

    # Read spectral bands
    for band in range(1, 6):
        data, ndv, epsg, extent = nivapy.spatial.read_raster(raw_path, band_no=band)

        if equalise:
            data = img_as_ubyte(exposure.equalize_hist(data))

        data_dict[str(band)] = data.flatten()

    # Read manually classified data (1 band only)
    man_img, ndv, man_epsg, extent = nivapy.spatial.read_raster(man_path, band_no=1)
    data_dict["y"] = man_img.flatten()

    # Build df
    df = pd.DataFrame(data_dict)
    del data_dict

    # Remove NoData and Other
    if dropna:
        df = df.query("y > 0")

    # Tidy
    df.reset_index(inplace=True, drop=True)
    df = df[["y"] + [str(i) for i in range(1, 6)]]

    return df


def classification_report(truths, preds, class_labels, class_names):
    """Print some simple classification skill metrics."""
    print(
        "Classification report:\n%s"
        % metrics.classification_report(
            truths,
            preds,
            labels=class_labels,
            target_names=class_names,
        ),
    )

    print(
        "Classification accuracy: %f"
        % metrics.accuracy_score(
            truths,
            preds,
        ),
    )
