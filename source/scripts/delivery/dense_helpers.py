"""Frozen numerical helpers isolated from research artifact loaders."""
import numpy as np
import nibabel as nib
import torch

class DenseAdapter(torch.nn.Module):
    def __init__(self,core):
        super().__init__();self.encoder=core.encoder;self.decoder=core.decoder;self.segmenter=core.segmenter
    def inference_step(self,images,batch_num=0):
        features=self.decoder(self.encoder(images));seg=self.segmenter(features)
        probability=self.segmenter.postprocess_for_inference(seg)['pred_seg']
        # Preserve original native head probabilities; use FP32 for Gaussian accumulation.
        return {'pred_seg':probability[:,1:2].float()}

def normalized_image(path):
    image = nib.load(str(path)); arr = image.get_fdata(dtype=np.float32)
    assert np.isfinite(arr).all()
    sub = arr[::4, ::4, ::4]; sub = sub[sub != 0]; assert len(sub)
    low, high = np.percentile(sub, [.5, 99.5])
    arr -= float(low); arr /= max(float(high - low), 1e-6); np.clip(arr, 0, 1, out=arr)
    return arr, image.affine, {'low_p005': float(low), 'high_p995': float(high), 'upper_clip': 1.}
