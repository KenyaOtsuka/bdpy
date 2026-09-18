# Features and DecodedFeatures

bdpy provides classes to handle DNN's (true) features and decoded features: `dataform.Features` and `dataform.DecodedFeatures`.

## Basic usage

``` python
from bdpy.dataform import Features, DecodedFeatures


## Initialize

features = Features('/path/to/features/dir')

decoded_features = DecodedFeatures('/path/to/decoded/features/dir')

## Get features as an array

feat = features.get(layer='conv1')

decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC', label='stimulus-0001)  # Decoded features for specified sample (label)
decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC')                        # Decoded features from all avaiable samples

# Decoded features with CV
decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC', fold='cv_fold1)

## List labels

feat_labels = features.labels

decfeat_labels = decoded_features.labels          # All available labels
decfeat_labels = decoded_features.selected_label  # Labels assigned to decoded features previously obtained by `get` method
```

## Feature statistics

``` python
features.statistic('mean', layer='fc8')
features.statistic('std', layer='fc8')          # Default ddof = 1
features.statistic('std, ddof=0', layer='fc8')

decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC')
decoded_features.statistic('std', layer='fc8', subject='sub-01', roi='VC')          # Default ddof = 1
decoded_features.statistic('std, ddof=0', layer='fc8', subject='sub-01', roi='VC')

# Decoded features with CV
decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC', fold='cv_fold1')  # Mean within the specified fold
decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC')

# If `fold` is omitted for CV decoded features, decoded features are pooled across add CV folds and then the statistics are calculated.

```

## Chunked HDF5 feature storage

The layout above stores one file per stimulus, so the only file boundary is the
sample axis: reading a few channels of a layer still costs a full read of every
stimulus. For large DNN features, `Features` can instead read **chunked HDF5**
storage, one file per layer:

```
features/
  conv1_1.h5    # /features (n_stimuli, *feature_shape) + /labels (n_stimuli,)
  conv2_1.h5
  fc8.h5
```

`/features` is explicitly chunked along both the sample axis and the outermost
feature axis, so a slice reads only the chunks it covers.

### Reading

Nothing changes for existing code -- `Features` detects the layout per
directory, and `.mat` trees keep working exactly as before:

``` python
features = Features('/path/to/features')   # either layout
feat = features.get(layer='conv5')
```

Pass `format='mat'` or `format='hdf5'` to skip detection. A directory holding
both layouts is read as `.mat` unless you say otherwise.

### Partial reads

`feature_slice` selects along the feature axes (axis 1 and up). On chunked HDF5
this is a real partial read; on the legacy layout the files are loaded in full
and then sliced, so the same code works on both.

``` python
import numpy as np

# Channels 128-255 only, without loading the rest of the layer
feat = features.get(layer='conv5', feature_slice=np.s_[128:256])

# Combine with label selection; rows come back in the order given
feat = features.get(
    layer='conv5',
    label=['stimulus-0003', 'stimulus-0001'],
    feature_slice=np.s_[128:256],
)

# Full shape without reading anything
n_stimuli, *feature_shape = features.shape('conv5')
```

To stream a layer that does not fit in memory, iterate. `iter_chunks` yields
`(slice, block)` so results can be placed back without tracking offsets, and by
default uses the on-disk chunk extent, so every element is read exactly once:

``` python
out = np.empty(features.shape('conv5'))
for sl, block in features.iter_chunks('conv5', axis=1):
    out[:, sl] = transform(block)
```

`axis=0` iterates over stimuli instead of features.

### Writing

Write a whole layer at once:

``` python
from bdpy.dataform import save_features

save_features('features/conv5.h5', array, labels)
```

Or incrementally, which is what feature extraction needs since it produces one
stimulus at a time:

``` python
from bdpy.dataform import FeatureWriter

with FeatureWriter('features/conv5.h5', feature_shape=(256, 13, 13),
                   dtype=np.float32) as writer:
    for label, feature in extract():
        writer.append(feature, label)
```

### Migrating existing features

``` python
from bdpy.dataform import convert_features_to_hdf5

convert_features_to_hdf5('/path/to/features_mat', '/path/to/features_h5')
```

The converter reads through the same backend `Features` uses for the legacy
layout, and streams in batches, so a layer is never held in memory in full.

### Chunk shape

Chunk shape, not the file format alone, is what decides the cost of a partial
read: HDF5 reads whole chunks even when only a few of their elements are
selected. By default bdpy derives it from a 1 MiB budget, splitting it between
the sample axis and the outermost feature axis and keeping trailing spatial axes
whole -- e.g. `(1200, 256, 13, 13)` float32 becomes chunks of
`(39, 39, 13, 13)`.

That means reading a *single* stimulus costs one chunk row (~39 stimuli of I/O),
which matters for `FeaturesDataset`-style per-sample access. Tune it if your
access pattern is skewed:

``` python
save_features(path, array, labels, target_chunk_bytes=256 * 1024)  # smaller chunks
save_features(path, array, labels, chunks=(1, 256, 13, 13))        # per-stimulus
```

Compression is off by default, since every compressed chunk costs a
decompression on the way out. Pass `compression='gzip'` or `compression='lzf'`
when size matters more than read speed.

### Format

Files are marked with root attributes `bdpy_format="features"` and
`bdpy_format_version=1`, and are rejected on read if those are missing or
newer than the running bdpy understands.
