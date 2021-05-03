import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import json
from json.decoder import JSONDecodeError

from typing import List
import torch
import numpy as np


#helper
clamp = np.vectorize(lambda x: min(max(x,0.),1.))

# helper
def arrayToBase64IM(samp):

    handle = io.BytesIO()
    channels = samp.shape[0]
    if channels == 1:
        plt.imsave(handle,samp.squeeze(0))
    elif channels == 3:
        plt.imsave(handle,clamp(torch.Tensor(samp).permute([1,2,0]))) # :(
    else:
        raise NotImplementedError

    handle.seek(0)
    image_encoded = base64.b64encode(handle.read())

    return image_encoded.decode('utf-8')

#helper
def get_entries(lines: List[str]):

    entries = []
    for line in lines:

        try:
            line_dict = json.loads(line[(line.find(' - ') + 3):])
            entries.append(line_dict)
        except JSONDecodeError:
            pass

    print(f"got {len(entries)} json entries")
    return entries



def renderLogAsHTML(logfile):
    """Returns html displaying images from logfile

    Currently looks for lines generated from processors.influence_transform
    """

    IF_PATTERN = {"true_label",
                  "pred_label",
                   "sample",
                   "confidence",
                   "closest_label",
                   "closest_sample"}

    with open(logfile) as f:
        dict_entries = get_entries(f.readlines())

    influence_entries = [d for d in dict_entries if IF_PATTERN.issubset(d.keys())]

    html_return = ""

    for entry in influence_entries:
        html_string = f"""
        <p>Test sample with label {entry["true_label"]}, predicted as {entry["pred_label"]}.
        Confidence in true label is {entry["confidence"]}</p>
        <div><img width=100 src="data:image/png;charset=utf-8;base64, {entry["sample"]}" /></div>
        <hr>
        """

        for id, (cl, cs) in enumerate(zip(entry["closest_label"],entry["closest_sample"])):

            closest_entry = f"""
            <p>Corresponding label Number {id} is {cl}. Corresponding sample is:</p>
            <div><img width=100 src="data:image/png;charset=utf-8;base64, {cs}" /></div>
            """
            html_string += closest_entry

        html_return += html_string + "<hr><br>"


    return html_return
