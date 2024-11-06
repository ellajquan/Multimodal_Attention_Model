from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import numpy as np

def read_mg(file_paths, img_rows, img_cols, as_gray, channels):
    images = []
    for file_path in file_paths:
        # Load the image
        image = imread(file_path, as_gray=as_gray)
        
        # If grayscale and expecting RGB, convert to 3 channels
        if as_gray and channels == 3:
            image = gray2rgb(image)  # Converts to (height, width, 3)
        # Resize the image to the specified dimensions
        image = resize(image, (img_rows, img_cols), anti_aliasing=True)
        
        images.append(image)
    
    # Convert to numpy array and normalize
    images = np.asarray(images, dtype=np.float32)
    images = (images - images.min()) / (images.max() - images.min())
    
    print(f"Shape before reshape: {images.shape}")  # Debugging print
    images = images.reshape(images.shape[0], img_rows, img_cols, channels)
    return images
   
def read_us(file_paths, img_rows, img_cols, as_gray, channels):
    """
  Reads the image files (ultrasound) and normalize the pixel values
    @params:
      file_paths - Array of file paths to read from
      img_rows - The image height.
      img_cols - The image width.
      as_grey - Read the image as Greyscale.
      channels - Number of channels.   
    """
    images=[]
  
    for file_path in file_paths:
      
        images.append(imread(file_path,  as_gray))
  
    images = np.asarray(images, dtype=np.float32)
    images = np.stack((images,)*3, axis=-1)
    images = (images-images.min())/(images.max()-images.min())
    images = images.reshape(images.shape[0], img_rows, img_cols, channels)
    return images