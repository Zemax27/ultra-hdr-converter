1. Requirements
Ensure you have the full version of imagecodecs installed to include the cms and ultrahdr extensions:

Bash
pip install -U "imagecodecs[all]"
2. The Implementation Workflow
Phase A: Decode and Linearize via ICC Profile
To linearize an image according to its embedded ICC profile, you use the cms (Color Management System) module within imagecodecs.

Python
import imagecodecs
import numpy as np

# 1. Decode JPEG and extract ICC profile
with open('input.jpg', 'rb') as f:
    jpeg_bytes = f.read()

# Decode the image to a NumPy array
sdr_array = imagecodecs.jpeg_decode(jpeg_bytes)

# Extract the ICC profile from the JPEG metadata
# imagecodecs.jpeg_metadata returns a dictionary of segments/tags
metadata = imagecodecs.jpeg_metadata(jpeg_bytes)
icc_bytes = metadata.get('icc_profile')

if icc_bytes:
    # 2. Create CMS profiles
    # Source: The embedded ICC profile
    src_profile = imagecodecs.cms_profile(icc_bytes)
    
    # Destination: A linearized version of the same profile
    # The 'linear=True' flag creates a profile with gamma 1.0 using the same primaries
    dst_profile = imagecodecs.cms_profile(icc_bytes, linear=True)
    
    # 3. Transform the array to linear space
    # This converts the pixels (e.g., from sRGB/Display-P3 curve to Linear)
    linear_sdr = imagecodecs.cms_transform(sdr_array, src_profile, dst_profile)
else:
    print("No ICC profile found; assuming standard sRGB.")
    linear_sdr = imagecodecs.cms_transform(sdr_array, 'srgb', 'srgb', linear=True)
Phase B: Encode Ultra HDR with Custom Gain Map
The ultrahdr_encode function in imagecodecs wraps libultrahdr. To encode using a pre-existing gain map and an SDR base, you pass both arrays. The library will handle the packaging into the MPF (Multi-Picture Format) container.

Python'''
# Assume you have your processed arrays:
# sdr_base: uint8 NumPy array (SDR image)
# gain_map: uint8 NumPy array (The gain map)

# Encode to Ultra HDR JPEG
# Note: You can pass the original 'icc_bytes' to preserve color management 
# for the SDR portion of the Ultra HDR file.
ultrahdr_bytes = imagecodecs.ultrahdr_encode(
    sdr_base, 
    gainmap=gain_map,
    # Many versions of the encoder allow embedding metadata via bytes
    metadata={'icc_profile': icc_bytes} 
)

with open('output_ultrahdr.jpg', 'wb') as f:
    f.write(ultrahdr_bytes)'''

🗝️ Critical Technical Notes
Linearization Precision: When you linearize to linear_sdr, the resulting values often exceed the uint8 range or require higher precision to avoid banding. I recommend setting outdtype=np.float32 in cms_transform if you plan to do further math on the linear data.

Gain Map Standards: In the Ultra HDR (v1.0/v1.1) spec, the gain map is typically a single-channel (grayscale) 8-bit image representing the log-ratio between SDR and HDR. If your gain_map array is 3-channel (RGB), imagecodecs will encode it as a multi-channel gain map.

CICP Tags: While ICC profiles are great for legacy SDR, Ultra HDR heavily relies on CICP (Coding-Independent Code Points) for the HDR reconstruction. If imagecodecs detects an ICC profile, it will attempt to map those primaries to the nearest CICP equivalent (e.g., BT.709 or P3) to satisfy the libultrahdr requirements.