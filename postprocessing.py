"""Output contract validation; learned postprocessing remains in source/scripts/."""
import numpy as np
import SimpleITK as sitk

def aligned_output(prediction, reference):
    if prediction.GetDimension() != 3 or reference.GetDimension() != 3:
        raise ValueError("Expected scalar 3D images")
    if prediction.GetSize() != reference.GetSize():
        raise ValueError("Prediction dimensions differ from input")
    if prediction.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected scalar label image")
    for attr in ("GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(prediction, attr)(), getattr(reference, attr)(), rtol=0, atol=1e-5):
            raise ValueError("Prediction geometry differs: " + attr)
    corners = np.stack(np.meshgrid(*[[0, n-1] for n in reference.GetSize()], indexing="ij")).reshape(3,-1).T
    for c in corners:
        index=tuple(map(int,c))
        if np.linalg.norm(np.array(prediction.TransformIndexToPhysicalPoint(index))-reference.TransformIndexToPhysicalPoint(index)) > 1e-4:
            raise ValueError("Prediction physical corner displacement exceeds 1e-4 mm")
    a = sitk.GetArrayViewFromImage(prediction)
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 52) or np.any(a != np.floor(a)):
        raise ValueError("Expected integer labels 0..52")
    result = sitk.Cast(prediction, sitk.sitkUInt8)
    # Remove only NIfTI header rounding after validating physical alignment.
    result.CopyInformation(reference)
    return result
