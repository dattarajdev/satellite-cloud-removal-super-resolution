# Super-resolve one Sentinel-2 TIFF

1. Drag a GeoTIFF onto `Run Super Resolution.bat`.
2. Wait for the run to finish.
3. Find the result in the `outputs` folder as `<input-name>_sr.tif`.

Double-clicking the launcher opens a file picker instead.

The TIFF can be a four-band RGBN file or a complete Sentinel-2 multiband TIFF.
Files with `B4`, `B3`, `B2`, and `B8` band names are reordered automatically,
regardless of their order. An unnamed four-band TIFF is assumed to already be
in B4, B3, B2, B8 order; an unnamed 13-band TIFF is read using the standard
Sentinel-2 ordering. Values stored as 0-1 reflectance are automatically scaled
to 0-10000. RGB-only TIFFs are stopped with a clear message because NIR cannot
be reconstructed reliably.

Map coordinates are retained when the input is georeferenced.
