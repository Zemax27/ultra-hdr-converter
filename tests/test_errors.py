from ultra_hdr_converter.errors import (
    AlreadyUltraHDRError,
    ColorTransformError,
    GainMapConfigError,
    GainMapDimensionError,
    GainMapError,
    GainMapShapeMismatchError,
    JpegStructureError,
    UltraHdrError,
)


def test_exception_hierarchy():
    assert issubclass(GainMapError, UltraHdrError)
    assert issubclass(GainMapShapeMismatchError, GainMapError)
    assert issubclass(GainMapDimensionError, GainMapError)
    assert issubclass(GainMapConfigError, GainMapError)
    assert issubclass(ColorTransformError, UltraHdrError)
    assert issubclass(JpegStructureError, UltraHdrError)
    assert issubclass(AlreadyUltraHDRError, UltraHdrError)


def test_gain_map_shape_mismatch_error_stores_shapes():
    err = GainMapShapeMismatchError(gain_map_shape=(2, 2), sdr_shape=(4, 4, 3))
    assert err.gain_map_shape == (2, 2)
    assert err.sdr_shape == (4, 4, 3)
    assert "(2, 2)" in str(err)
    assert "(4, 4)" in str(err)
