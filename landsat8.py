from os import path
from osgeo import gdal, osr, gdal_array, ogr
from datetime import datetime

import logging
import numpy as np

METADATA_FILE_SUFFIX = "_MTL.txt"
BAND_DATA_FILE_SUFFIX_TEMPLATE = "_B%i.TIF"
N_BANDS = 11  # Landsat 8 image bands

CALIBRATED_BAND_DTYPE = np.float32

logger = logging.getLogger(__name__)


class Landsat8Error(Exception):
    pass


class Landsat8L1T:
    """Landsat 8 OLI/TIRS sensor data"""

    def __init__(self, data_dir=''):
        if not path.exists(data_dir):
            raise Landsat8Error('Path does not exist: %s' % data_dir)
        if not path.isdir(data_dir):
            raise Landsat8Error('Directory with Landsat 8 data expected: %s' % data_dir)

        self.data_dir = path.abspath(data_dir)
        dirname = path.basename(self.data_dir)

        self.datasets = [None] * 11
        self.bands = [None] * 11
        self.geo_info = [None] * 11

        self.bands_data_files = [
            path.join(self.data_dir, dirname + BAND_DATA_FILE_SUFFIX_TEMPLATE % b)
            for b in range(1, N_BANDS+1)
        ]
        metadata_fname = path.join(self.data_dir, dirname + METADATA_FILE_SUFFIX)
        self.metadata = load_metadata(metadata_fname)

    def get_dataset(self, band_nr, mode=gdal.GA_ReadOnly):
        """Return a GDAL DataSet with the corresponding Band info."""

        b = band_nr - 1
        if self.datasets[b] is None:
            self.datasets[b] = gdal.Open(self.bands_data_files[b], mode)
        return self.datasets[b]

    def get_band_data(self, band_nr):
        """Return the given band's data as a Numpy array."""

        b = band_nr - 1
        if self.bands[b] is None:
            raw_data = self.get_dataset(band_nr)
            self.bands[b] = raw_data.ReadAsArray()

        return self.bands[b]

    def get_geo_info(self, band_nr):
        b = band_nr - 1
        if self.geo_info[b] is None:
            dataset = self.get_dataset(band_nr)
            self.geo_info[b] = get_geo_info(dataset)
        return self.geo_info[b]

    def calibrate_band(self, band_nr):
        """Convert the band's NDs to Top Of Atmosphere (TOA) Radiance. Return a new array."""
        # ND to TOA Radiance: http://landsat.usgs.gov/Landsat8_Using_Product.php
        # L = MQ + A
        # L = TOA spectral radiance (Watts/( m2 * srad * μm))
        # M = Band-specific multiplicative rescaling factor from the metadata
        #     (RADIANCE_MULT_BAND_x, where x is the band number)
        # A = Band-specific additive rescaling factor from the metadata
        #     (RADIANCE_ADD_BAND_x, where x is the band number)
        # Q = Quantized and calibrated standard product pixel values (DN)\
        data = self.get_band_data(band_nr)
        radiometric_calibration_paramas = self.metadata['RADIOMETRIC_RESCALING']
        multiplicative_factor = radiometric_calibration_paramas['RADIANCE_MULT_BAND_%i' % band_nr]
        additive_factor = radiometric_calibration_paramas['RADIANCE_ADD_BAND_%i' % band_nr]
        return calibrate(data, multiplicative_factor, additive_factor)

    def save(self, band_nr, calibrate=False):
        """
        Write the bands data to disk.

        If single_file is True (default) create a single, multi band, GEOTIFF file.

        """
        # TODO: create a new (updated) metadata file.

        if calibrate:
            data = self.calibrate_band(band_nr)
            suffix = ".calibrated.tiff"
        else:
            data = self.get_band_data(band_nr)
            suffix = ".test.tiff"

        driver = gdal.GetDriverByName('GTiff')
        geo_transform, projection = self.get_geo_info(band_nr)
        output_fname = path.basename(self.bands_data_files[band_nr-1]) + suffix
        rows, cols = data.shape
        data_type = {t: v for v, t in gdal_array.codes.items()}[data.dtype.type]
        dataset = driver.Create(output_fname, cols, rows, 1, data_type)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(data)
        dataset = None  # Closes the file
        return output_fname

    def create_mask_from_data(self, data, dataset_name='', dataset_format='MEM', geo_from_band=1):
        driver = gdal.GetDriverByName(dataset_format)
        geo_transform, projection = self.get_geo_info(geo_from_band)
        band = self.get_dataset(geo_from_band).GetRasterBand(1)
        target_ds = driver.Create(dataset_name, band.XSize, band.YSize, 1, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(geo_transform)
        target_ds.SetProjection(projection.ExportToWkt())
        band = target_ds.GetRasterBand(1)
        band.WriteArray(data)
        band.FlushCache()
        return target_ds

    def create_mask_file_from_data(self, data, geo_from_band=1, suffix="mask"):
        output_fname = path.basename(self.bands_data_files[geo_from_band]) + '.%s.tiff' % suffix
        target_ds = self.create_mask_from_data(self, data, dataset_name=output_fname,
                                               dataset_format='GTiff')
        target_ds = None  # Closes the file
        return output_fname


def load_metadata(metadata_path):
    """Load a _MTL.txt file as a dict."""

    def parse_value(val):
        val = val.strip()
        if val.startswith('"'):
            return val.replace('"', '')
        elif val.isdigit():
            return int(val)
        elif '-' in val and ':' not in val and '.' not in val:
            return datetime.strptime(val, '%Y-%m-%d')
        elif '-' in val and ':' in val:
            return datetime.strptime(val, '%Y-%m-%dT%H:%M:%SZ')
        else:
            return float(val)

    raw_metadata = open(metadata_path).readlines()
    metadata = {}
    current_dict = metadata
    previous_dict = metadata
    for line in raw_metadata:
        line = line.strip()
        if line.startswith('GROUP'):
            _, title = line.split(' = ')
            previous_dict = current_dict
            current_dict = {}
            previous_dict[title] = current_dict
        elif line.startswith('END'):
            current_dict = previous_dict
        else:
            key, val = line.split(' = ')
            current_dict[key] = parse_value(val)
    return metadata['L1_METADATA_FILE']


def get_geo_info(dataset):
    """Function to read a file's projection"""

    geo_transform = dataset.GetGeoTransform()
    projection = osr.SpatialReference()
    projection.ImportFromWkt(dataset.GetProjectionRef())
    return (geo_transform, projection)


def calibrate(data, multiplicative_factor, additive_factor):
    calibrated_data = data.astype(CALIBRATED_BAND_DTYPE)
    calibrated_data *= multiplicative_factor
    calibrated_data += additive_factor
    return calibrated_data


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection, target_value=1,
                            output_fname='', dataset_format='MEM'):
    """
    Rasterize the given vector (wrapper for gdal.RasterizeLayer). Return a gdal.Dataset.
    :param vector_data_path: Path to a shapefile
    :param cols: Number of columns of the result
    :param rows: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    :param target_value: Pixel value for the pixels. Must be a valid gdal.GDT_UInt16 value.
    :param output_fname: If the dataset_format is GeoTIFF, this is the output file name
    :param dataset_format: The gdal.Dataset driver name. [default: MEM]
    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """
    Rasterize, in a single image, all the vectors in the given directory.
    The data of each file will be assigned the same pixel value. This value is defined by the order
    of the file in file_paths, starting with 1: so the points/poligons/etc in the same file will be
    marked as 1, those in the second file will be 2, and so on.
    :param file_paths: Path to a directory with shapefiles
    :param rows: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    labeled_pixels = np.zeros((rows, cols))
    reference = {}
    for i, path in enumerate(file_paths):
        label = i + 1
        logger.debug("Processing file %s: label (pixel value) %i", path, label)
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        reference[path] = label
        band = ds.GetRasterBand(1)
        aux = band.ReadAsArray()
        logger.debug("Labeled pixels: %i", len(aux.nonzero()[0]))
        labeled_pixels += aux
        ds = None
    return {
        'reference': reference,
        'raster': labeled_pixels
    }


# A list of "random" colors
COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]


def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    ct = gdal.ColorTable()

    for pixel_value in range(len(COLORS)):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'Machinalis',
        'TIFFTAG_IMAGEDESCRIPTION': 'Machinalis Labs Experiment',
        'TIFFTAG_MAXSAMPLEVALUE': str(int(data.max())),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    band.WriteArray(data)

    dataset = None  # Close the file
    return
