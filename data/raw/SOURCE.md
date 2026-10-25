# MNIST raw data — source and verification

These 4 files are the original MNIST handwritten digit dataset (60,000
training images + 10,000 test images, 28x28 grayscale, in the standard
idx-ubyte format), downloaded on 2026-10-17.

Mirror used: https://github.com/golbin/TensorFlow-MNIST (`mnist/data/`),
which hosts the same files Yann LeCun's original site and
`tf.keras.datasets.mnist.load_data()` serve — this environment could not
reach `storage.googleapis.com` or `yann.lecun.com` directly, so a GitHub
mirror was used instead and the result was verified byte-for-byte.

## Integrity check (MD5)

| File                          | MD5                              | Matches official |
|--------------------------------|-----------------------------------|:---:|
| train-images-idx3-ubyte.gz     | f68b3c2dcbeaaa9fbdd348bbdeb94873  | ✅ |
| train-labels-idx1-ubyte.gz     | d53e105ee54ea40749a09fcbcd1e9432  | ✅ |
| t10k-images-idx3-ubyte.gz      | 9fb629c4189551a2d022fa330f9573f3  | ✅ |
| t10k-labels-idx1-ubyte.gz      | ec29112dd5afa0611ce80d1b7f02629c  | ✅ |

These are the official checksums for the MNIST dataset — confirming these
are the genuine, complete files, not a reduced or synthetic substitute.

`src/data_loader.py` reads these files directly (parsing the idx format
itself), so the project does not need internet access to train or
evaluate. If this folder is ever missing (e.g. a fresh clone without the
data files), `data_loader.py` automatically falls back to
`tf.keras.datasets.mnist.load_data()`, which will download MNIST itself
on a machine that does have internet access (such as Google Colab).
