# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from torch.hub import load_state_dict_from_url
import argparse
import torch

from compressai.zoo import load_pretrained

root_url = "https://nextcloud.isae.fr/index.php/s/"
model_urls = {
    "mbt": {
        "mse": {
            "0.0035": f"{root_url}/3qkx8YpcS87z5QW/download/mbt_0.0035_bsd.pth.tar",
            "0.013": f"{root_url}/extEnwSdd5gLNRA/download/mbt_0.013_bsd.pth.tar",
            "0.0483": f"{root_url}/KCCaBoNjGcqKSsF/download/mbt_0.0483_bsd.pth.tar",
            "0.1800": f"{root_url}/eFxMeJZ6Az5T9B9/download/mbt_0.1800_bsd.pth.tar",
        },
    },
    "cheng": {
        "mse": {
            "0.0035": f"{root_url}/GXxykHwX4keGYWs/download/cheng_0.0035_bsd.pth.tar",
            "0.013": f"{root_url}/iAwLZwCdSrBCEtA/download/cheng_0.013_bsd.pth.tar",
            "0.0483": f"{root_url}/LwsSyyBCRHc74Mg/download/cheng_0.0483_bsd.pth.tar",
            "0.1800": f"{root_url}/pSM8LdS2bksfFG9/download/cheng_0.1800_bsd.pth.tar",
        },
    },
}

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, choices=["mbt", "cheng"], help="model type")
parser.add_argument('--bitrate', type=float, help="Determines the model bitrate. Choices are 0.0035, 0.013, 0.0483, 0.1800")
parser.add_argument('--save_path', type=str, default="model_zoo/", help="path to save the checkpoint")

args = parser.parse_args()

url = model_urls[args.model_type]["mse"][args.quality]
state_dict = load_state_dict_from_url(url, progress=True)
state_dict = load_pretrained(state_dict)

if not os.path.exists(os.path.dirname(args.save_path)):
    os.makedirs(args.save_path)

torch.save({"state_dict": state_dict}, args.save_path)
