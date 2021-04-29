import matplotlib.pyplot as plt
import io
import base64
from PIL import Image



def arrayToBase64IM(samp):

    handle = io.BytesIO()
    plt.imsave(handle,samp.squeeze(0))
    handle.seek(0)
    image_encoded = base64.b64encode(handle.read())

    return image_encoded.decode('utf-8')
