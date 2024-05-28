from osgeo import gdal

import poldecomp as pd

band = gdal.Open("C:\S1B_IW_SLC__1SDV_20210811T230839_20210811T230907_028205_035D6F_21CD\S1B_IW_SLC__1SDV_20210811T230839_20210811T230907_028205_035D6F_21CD.SAFE\measurement\s1b-iw1-slc-vh-20210811t230839-20210811t230907-028205-035d6f-001.tiff")
decomposition = pd.General4SD(band, 5)
decomposition.get_result()

