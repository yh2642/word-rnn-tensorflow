from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

sample_user = {
'yhu': 'category.55cdd55f830f780e86a031ad category.56a6eabe8e7989cd56fcea61 category.5590e712d48bac2951b9bbe2 category.569701c240a25b5bd66883d0 category.5590e713d48bac2951b9bc19 category.56a6e2258e7989cd56fcea31 category.5697023540a25b5bd66883d2 category.5590e712d48bac2951b9bbe9 category.55cdd55f830f780e86a03246 category.56a6eabe8e7989cd56fcea61 category.56a6e8cf8e7989cd56fcea58 category.55b6740de7798944b4792573 category.55cdd55e830f780e86a03167 category.56a6e7608e7989cd56fcea51 category.55cdd55e830f780e86a03167 category.5590e713d48bac2951b9bc19 category.55cdd55e830f780e86a0313a category.56a6eabe8e7989cd56fcea61 category.569701c240a25b5bd66883d0 category.56a6eabe8e7989cd56fcea61 category.57fa069a805d8955dcc01012 category.572dd173939e250d08ec59f3 category.55b6740de7798944b4792501 category.572dd172939e250d08ec593c category.572dd173939e250d08ec59c9 category.55b6740de7798944b47924ff category.56a6eabe8e7989cd56fcea61 category.56a6de2f8e7989cd56fcea22 category.572dd172939e250d08ec5946 category.54f19623b4c4d660e6f18a96 category.56a6eabe8e7989cd56fcea61 category.55cdd55e830f780e86a03167 category.5731c8be939e254948159bbb category.56a6eabe8e7989cd56fcea61 category.55cdd55e830f780e86a03167 category.5590e713d48bac2951b9bc19 category.57e51518805d8905e38308e4 category.54f186dcb4c4d616d0cf2289 category.5697011040a25b5bd66883cd category.572dd171939e250d08ec58ba category.55cdd55f830f780e86a031cf category.5697041140a25b5bd66883d9 category.56a6e74a8e7989cd56fcea50 category.55cdd55e830f780e86a03167 category.56a6e8cf8e7989cd56fcea58 category.572d9ebf561feb5795a50df9 category.56a6e2258e7989cd56fcea31 category.56a6e82c8e7989cd56fcea53 category.5590e713d48bac2951b9bc19 category.56a6eabe8e7989cd56fcea61 category.572dd172939e250d08ec594a category.572dd172939e250d08ec594a category.569701c240a25b5bd66883d0 category.57bd8e7f805d896b0686cd77 category.589c65f8805d891801d314fe category.56a6eabe8e7989cd56fcea61 category.55b6740de7798944b4792573 category.55b6740de7798944b479252d category.57e51518805d8905e38308e4 category.5590e713d48bac2951b9bc19 category.58ca004a805d891d72d2ef4d category.56a6e84c8e7989cd56fcea56 category.56a6e82c8e7989cd56fcea53 category.55b6740de7798944b4792573 category.55b6740de7798944b4792545 category.55b6740de7798944b4792545 category.55cdd55e830f780e86a03167 category.56a6e8cf8e7989cd56fcea58 category.55b6740de7798944b4792566 category.55cdd55f830f780e86a03201 category.56a6e8cf8e7989cd56fcea58 category.5590e713d48bac2951b9bc14 category.5590e713d48bac2951b9bc14 category.5590e713d48bac2951b9bc14 category.589c65f8805d891801d314fe category.54f19391b4c4d660e6f18a93 category.572dd172939e250d08ec593e category.57e51519805d8905e3830922 category.572dd172939e250d08ec593c category.56a6eabe8e7989cd56fcea61 category.55b6740de7798944b4792545 category.56a6ea7a8e7989cd56fcea5f category.55b6740de7798944b4792545 category.55b6740de7798944b4792545 category.55cdd55f830f780e86a03201 category.55cdd55f830f780e86a03201 category.55b6740de7798944b4792545 category.56a6e8cf8e7989cd56fcea58 category.55cdd55f830f780e86a031ae category.54f186e0b4c4d616d0cf228a category.55b6740de7798944b4792573 category.54f18719b4c4d616d5cf2273 category.55cdd55f830f780e86a031d2 category.56937aa33a22e443b2330748 category.5590e713d48bac2951b9bc19 category.55cdd55e830f780e86a03167 category.56a6eabe8e7989cd56fcea61 category.55cdd55e830f780e86a03167 category.54f18739b4c4d616d5cf2274 category.572dd171939e250d08ec58d3 category.574962cc939e2554fa18494d category.574962cc939e2554fa18494d category.572dd171939e250d08ec5921 category.55cdd55f830f780e86a031cf category.572d9ebf561feb5795a50df9 category.55cdd55f830f780e86a031ae category.55cdd55e830f780e86a03185 category.572dd171939e250d08ec58b3 category.57e51518805d8905e38308e4 category.56a6dce48e7989cd56fcea1d category.55b6740de7798944b4792573 category.5590e713d48bac2951b9bc19 category.55cdd55e830f780e86a03167 category.572dd171939e250d08ec591e category.572dd171939e250d08ec58ef category.572dd172939e250d08ec599e category.572dd171939e250d08ec5938 category.55b6740de7798944b4792573 category.56a6eabe8e7989cd56fcea61 category.5732a0f7939e2568aecd4bd2 category.56a6e52b8e7989cd56fcea44 category.582d4db9805d89419841331b category.582d4db9805d89419841331b category.54f198d2b4c4d660e7f18abb category.56a6dbcd8e7989cd56fcea12 category.582d4db9805d89419841331b category.572dd172939e250d08ec594a category.55cdd55f830f780e86a031a4 category.572dd173939e250d08ec59d5 category.55cdd55f830f780e86a0321b category.56a6e84c8e7989cd56fcea56 category.582d4db9805d89419841331b category.5590e713d48bac2951b9bc14 category.57bd8e7a805d896b0686cd40 category.574962cc939e2554fa18494d category.57e51518805d8905e38308e4 category.572dd171939e250d08ec58ef category.572dd172939e250d08ec597b category.572dd171939e250d08ec58d3 category.572dd171939e250d08ec58ba category.5590e712d48bac2951b9bbd1 category.56937c613a22e443b2330758 category.55cdd55f830f780e86a03207 category.55cdd55e830f780e86a03169 category.5590e712d48bac2951b9bbc4 category.57e51518805d8905e38308e1 category.572dd172939e250d08ec5948 category.55cdd55e830f780e86a03130 category.55cdd55e830f780e86a03131 category.573fe387939e257b45dc7735 category.589c65f8805d891801d314fe category.54f198d2b4c4d660e7f18abb category.54f198d2b4c4d660e7f18abb category.572dafe2561feb5795a50e0f category.54f18719b4c4d616d5cf2273 category.572dafe2561feb5795a50e0f category.54f18719b4c4d616d5cf2273',
'hua': 'category.57bd8e7f805d896b0686cd73 category.54f184dcb4c4d616d1cf226b category.589c65f8805d891801d314fe category.54f186dcb4c4d616d0cf2289 category.5590e713d48bac2951b9bc14 category.57bd8e7f805d896b0686cd76 category.55cdd55e830f780e86a03167 category.54f18719b4c4d616d5cf2273 category.572dafe2561feb5795a50e0f category.54f18739b4c4d616d5cf2274 category.54f18739b4c4d616d5cf2274 category.5590e713d48bac2951b9bc14 category.54f18719b4c4d616d5cf2273 category.5731c8bf939e254948159bbd category.56a6eabe8e7989cd56fcea61 category.572dafe2561feb5795a50e0f category.56a6f5b58e7989cd56fcea9e category.572dd171939e250d08ec58a5 category.572dd173939e250d08ec59f7 category.54f18577b4c4d616eccf22cc category.57bd8e7f805d896b0686cd73 category.572dd172939e250d08ec597b category.54f184c9b4c4d616d3cf2263 category.572dd172939e250d08ec5946 category.573fe388939e257b45dc773c category.572dd171939e250d08ec58ba category.572dd171939e250d08ec58ba category.573fe386939e257b45dc7733 category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc1a category.5697047f40a25b5bd66883da category.55cdd55f830f780e86a031cf category.54f19623b4c4d660e6f18a96 category.574962cc939e2554fa18495f category.572dd173939e250d08ec59f7 category.572dd171939e250d08ec58ba category.5590e713d48bac2951b9bc19 category.5590e712d48bac2951b9bbbf category.56a6f5a28e7989cd56fcea9b category.55cdd55e830f780e86a03167 category.572d9ebf561feb5795a50df9 category.54f1853db4c4d616eccf22c9 category.5590e713d48bac2951b9bc19 category.54f18719b4c4d616d5cf2273 category.572dd172939e250d08ec594a category.572dd172939e250d08ec594a category.55cdd55f830f780e86a031cf category.56a6eabe8e7989cd56fcea61 category.572dd171939e250d08ec58d6 category.56a6eabe8e7989cd56fcea61 category.569701c240a25b5bd66883d0 category.55b6740de7798944b47924f7 category.5590e713d48bac2951b9bc14 category.55cdd55f830f780e86a031b0 category.57e51518805d8905e38308e4 category.55b6740de7798944b47924f8 category.572dd173939e250d08ec59bb category.55cdd55e830f780e86a0313a category.55cdd55f830f780e86a031b0 category.572dd171939e250d08ec58ba category.5590e713d48bac2951b9bc14 category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc14 category.589c65f8805d891801d314fe category.589c65f8805d891801d314fe category.55b6740de7798944b47924ff category.56a6dc238e7989cd56fcea15 category.572dd171939e250d08ec5904 category.56a6e3608e7989cd56fcea37 category.573fe385939e257b45dc7730 category.573fe385939e257b45dc7730 category.55cdd55f830f780e86a031d2 category.55cdd55f830f780e86a031d2 category.56a6eabe8e7989cd56fcea61 category.54f18719b4c4d616d5cf2273 category.572dafe2561feb5795a50e0f category.572dd171939e250d08ec58d3 category.572dd172939e250d08ec599e category.54f18739b4c4d616d5cf2274 category.572dd171939e250d08ec5938 category.572dd172939e250d08ec599d category.55cdd55f830f780e86a031cf category.572d9ebf561feb5795a50df9 category.56a6da148e7989cd56fcea08 category.589c65f8805d891801d314fe category.572dd172939e250d08ec5979 category.55b6740de7798944b479257f category.572dd172939e250d08ec597b category.572dd171939e250d08ec58ef category.572dd171939e250d08ec58bf category.572dd171939e250d08ec58ef category.57e51518805d8905e38308e4 category.54f18719b4c4d616d5cf2273 category.56a6eabe8e7989cd56fcea61 category.55cdd55f830f780e86a031a2 category.572dd172939e250d08ec594a category.572dd172939e250d08ec594a category.572dd172939e250d08ec594a category.56a6e74a8e7989cd56fcea50 category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc1a category.56a6e84c8e7989cd56fcea56 category.55cdd55e830f780e86a03185 category.56a6eabe8e7989cd56fcea61 category.56a6eabe8e7989cd56fcea61 category.5590e713d48bac2951b9bc19 category.56a6e84c8e7989cd56fcea56 category.54f1853db4c4d616eccf22c9 category.55cdd55e830f780e86a03185 category.573fe386939e257b45dc7733 category.54f1853db4c4d616eccf22c9 category.5590e713d48bac2951b9bc14 category.57e51518805d8905e38308e4 category.573fe388939e257b45dc773c category.55cdd55f830f780e86a031b3 category.57e51518805d8905e38308dd category.54f186e0b4c4d616d0cf228a category.55b6740de7798944b479249e category.55cdd55f830f780e86a0324a category.572dd172939e250d08ec597b category.57bd8e7f805d896b0686cd74 category.572dd172939e250d08ec594a category.54f186e0b4c4d616d0cf228a category.57e51518805d8905e38308dd category.56a6eabe8e7989cd56fcea61 category.569700cf40a25b5bd66883cc category.5590e713d48bac2951b9bc14 category.57e51518805d8905e38308e1 category.57e51518805d8905e38308e4 category.572dd171939e250d08ec5921 category.55b6740de7798944b479257c category.55b6740de7798944b479257c category.56a6e0228e7989cd56fcea24 category.56a6de2f8e7989cd56fcea22 category.573fe387939e257b45dc7739 category.572dd173939e250d08ec59f7 category.55b6740de7798944b47924e5 category.572dd173939e250d08ec59cd category.572dd173939e250d08ec59cd category.572dd173939e250d08ec59cd category.572dd171939e250d08ec58ba category.572dd171939e250d08ec5938 category.572dd172939e250d08ec5969 category.56937c613a22e443b2330758 category.572dd173939e250d08ec5a00 category.55cdd55f830f780e86a031cf category.55cdd55f830f780e86a031cf category.55cdd55f830f780e86a031d2 category.55cdd55f830f780e86a031d2 category.55cdd55f830f780e86a031ad category.56a6d3048e7989cd56fce9e9 category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc16 category.572dd172939e250d08ec597b category.573fe386939e257b45dc7733 category.56a6eabe8e7989cd56fcea61 category.56a6e2578e7989cd56fcea33 category.56a6eabe8e7989cd56fcea61 category.54f18719b4c4d616d5cf2273 category.56a6eabe8e7989cd56fcea61 category.55cdd55e830f780e86a0313a category.56a6eabe8e7989cd56fcea61 category.56a6eabe8e7989cd56fcea61 category.56a6eabe8e7989cd56fcea61 category.55cdd55f830f780e86a031cc category.56a6e52b8e7989cd56fcea44 category.5697047f40a25b5bd66883da category.55cdd55f830f780e86a031ae category.56a6e2578e7989cd56fcea33 category.56a6e2578e7989cd56fcea33 category.572dd172939e250d08ec5955 category.572dd172939e250d08ec594a',
'yongqing': 'category.5731c8bf939e254948159bbd category.574962cc939e2554fa18494d category.574962cc939e2554fa184932 category.55b6740de7798944b479257c category.55cdd55e830f780e86a03135 category.55cdd55e830f780e86a03138 category.55b6740de7798944b47924f3 category.56a6f5b58e7989cd56fcea9e category.5590e712d48bac2951b9bbbf category.5590e712d48bac2951b9bbbf category.58ca004e805d891d72d2ef53 category.55cdd55e830f780e86a03167 category.56a6e83b8e7989cd56fcea55 category.57bd8e7f805d896b0686cd73 category.55cdd55e830f780e86a03135 category.55cdd55e830f780e86a03167 category.56a6eabe8e7989cd56fcea61 category.56a6eabe8e7989cd56fcea61 category.5697024340a25b5bd66883d3 category.55cdd55e830f780e86a03151 category.5590e712d48bac2951b9bbbf category.54f18577b4c4d616eccf22cc category.54f186e0b4c4d616d0cf228a category.5590e712d48bac2951b9bbbf category.54f18577b4c4d616eccf22cc category.5590e713d48bac2951b9bc19 category.5697036740a25b5bd66883d7 category.56a6eabe8e7989cd56fcea61 category.56a6e83b8e7989cd56fcea55 category.57bd8e7f805d896b0686cd73 category.56a6e83b8e7989cd56fcea55 category.572dd173939e250d08ec59d5 category.56a6eabe8e7989cd56fcea61 category.56a6eabe8e7989cd56fcea61 category.57bd8e7f805d896b0686cd73 category.56a6eabe8e7989cd56fcea61 category.56a6e8c68e7989cd56fcea57 category.57bd8e7f805d896b0686cd73 category.54f184dcb4c4d616d1cf226b category.572da80a561feb5795a50e08 category.5590e712d48bac2951b9bbb0 category.56a6e83b8e7989cd56fcea55 category.55cdd55e830f780e86a03185 category.56a6e83b8e7989cd56fcea55 category.55cdd55e830f780e86a03185 category.572dd171939e250d08ec58a5 category.55b6740de7798944b4792574 category.55b6740de7798944b4792573 category.5590e712d48bac2951b9bbbf',
'annie': 'category.54f18719b4c4d616d5cf2273 category.54f18719b4c4d616d5cf2273 category.54f184c9b4c4d616d3cf2263 category.57bd8e7f805d896b0686cd75 category.55b6740de7798944b47924f7 category.55cdd55e830f780e86a0315b category.55cdd55e830f780e86a0315b category.5590e712d48bac2951b9bbbf category.572dafd2561feb5795a50e0e category.55cdd55f830f780e86a03246 category.54f19cfeb4c4d660fcf18bb6 category.54f19cfeb4c4d660fcf18bb6 category.573fe385939e257b45dc7730 category.5590e713d48bac2951b9bc14 category.56a6f5b08e7989cd56fcea9d category.573fe385939e257b45dc7730 category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.56a6f6b78e7989cd56fceaa0 category.57bd8e7f805d896b0686cd75 category.57bd8e7f805d896b0686cd74 category.5590e712d48bac2951b9bbbe category.54f1853db4c4d616eccf22c9 category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.57bd8e7f805d896b0686cd73 category.56a6f6b78e7989cd56fceaa0 category.5590e713d48bac2951b9bc14 category.55cdd55f830f780e86a03246 category.572dd172939e250d08ec5946 category.574962cc939e2554fa18494d category.56a6e82c8e7989cd56fcea53 category.56a6e8cf8e7989cd56fcea58 category.56a6e82c8e7989cd56fcea53 category.54f18a17b4c4d616eccf231c category.56a6f6b78e7989cd56fceaa0 category.54f184dcb4c4d616d1cf226b category.572dd172939e250d08ec593c category.56a6f5b58e7989cd56fcea9e category.55b6740de7798944b47924ec category.574962cc939e2554fa18495a category.56a6e83b8e7989cd56fcea55 category.55b6740de7798944b4792564 category.56a6e83b8e7989cd56fcea55 category.5590e713d48bac2951b9bc14 category.57bd8e7f805d896b0686cd75 category.54f1868eb4c4d616eccf22db category.56a6e1608e7989cd56fcea2b category.56a6e20b8e7989cd56fcea2f category.56a6e3468e7989cd56fcea36 category.57bd8e7f805d896b0686cd73 category.55b6740de7798944b47924ec category.55b6740de7798944b47924ec category.54f19438b4c4d660f7f18acb category.55b6740de7798944b47924ec category.57bd8e7f805d896b0686cd74 category.56a6f5b58e7989cd56fcea9e category.572dafd2561feb5795a50e0e category.572dafd2561feb5795a50e0e category.55b6740de7798944b47924f8 category.55b6740de7798944b479256d category.56a6f5b58e7989cd56fcea9e category.54f184c9b4c4d616d3cf2263 category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc1a category.55cdd55e830f780e86a0315b category.54f186dcb4c4d616d0cf2289 category.54f186e0b4c4d616d0cf228a category.57e51519805d8905e38308f8 category.54f186dcb4c4d616d0cf2289 category.572dafd2561feb5795a50e0e category.5590e713d48bac2951b9bc14 category.55cdd55f830f780e86a03244 category.572dafe2561feb5795a50e0f category.56a6f6b78e7989cd56fceaa0 category.55b6740de7798944b4792564 category.56a6f5b58e7989cd56fcea9e category.55b6740de7798944b47924f8 category.57bd8e7f805d896b0686cd73 category.56a6f5b58e7989cd56fcea9e category.55cdd55e830f780e86a0312d category.55cdd55e830f780e86a0312d category.55cdd55e830f780e86a0315b category.572dd171939e250d08ec5911 category.56a6e2578e7989cd56fcea33 category.56a6e20b8e7989cd56fcea2f category.55cdd55e830f780e86a0315b category.54f18739b4c4d616d5cf2274 category.54f18719b4c4d616d5cf2273 category.54f18719b4c4d616d5cf2273 category.56a6f66a8e7989cd56fcea9f category.55b6740de7798944b4792573 category.54f19442b4c4d660fcf18ad9 category.56a6d9c18e7989cd56fcea05 category.55b6740de7798944b479256d category.55b6740de7798944b47924ec category.54f18957b4c4d616eecf22cd category.54f18957b4c4d616eecf22cd category.55cdd55e830f780e86a0315b category.5590e713d48bac2951b9bc14 category.5590e713d48bac2951b9bc1a category.54f186e0b4c4d616d0cf228a category.5590e712d48bac2951b9bbbe category.5590e712d48bac2951b9bbbe category.5590e712d48bac2951b9bbbe category.5590e712d48bac2951b9bbbe category.5590e712d48bac2951b9bbbe category.54f184c9b4c4d616d3cf2263 category.54f184dcb4c4d616d1cf226b category.57bd8e7f805d896b0686cd73 category.56a6f5b58e7989cd56fcea9e category.572dd173939e250d08ec59a3 category.5590e713d48bac2951b9bc0b category.54f184c9b4c4d616d3cf2263 category.56a6f6b78e7989cd56fceaa0 category.5590e712d48bac2951b9bbbe category.55b6740de7798944b47924ec category.57bd8e7f805d896b0686cd73 category.5590e712d48bac2951b9bbbe category.56a6e3468e7989cd56fcea36 category.55b6740de7798944b479256d category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc1a category.55b6740de7798944b4792573 category.573fe385939e257b45dc7730 category.572dd171939e250d08ec58fe category.573fe388939e257b45dc773a category.573fe388939e257b45dc773a category.574962cc939e2554fa184961 category.574962cc939e2554fa184961 category.5590e712d48bac2951b9bbe1 category.55b6740de7798944b4792573 category.56a6e2258e7989cd56fcea31 category.5590e712d48bac2951b9bbbe category.55cdd55e830f780e86a0315b category.54f1868eb4c4d616eccf22db category.56a6ea848e7989cd56fcea60 category.56a6f5b08e7989cd56fcea9d category.57e51518805d8905e38308df category.54f18719b4c4d616d5cf2273 category.55b6740de7798944b479257f category.572dafd2561feb5795a50e0e category.56a6dc9f8e7989cd56fcea1c category.5693616f0c9633abc535a58c category.54f18957b4c4d616eecf22cd category.54f1853db4c4d616eccf22c9 category.56a6d3048e7989cd56fce9e9 category.55cdd55e830f780e86a03167 category.5590e713d48bac2951b9bc14 category.5590e712d48bac2951b9bbae category.55cdd55e830f780e86a03167 category.57bd8e7f805d896b0686cd76 category.56a6d3048e7989cd56fce9e9 category.54f18739b4c4d616d5cf2274 category.54f18719b4c4d616d5cf2273 category.56a6f5a98e7989cd56fcea9c category.57bd8e7f805d896b0686cd77 category.55b6740de7798944b47924dc category.56a6f5b08e7989cd56fcea9d category.572dd173939e250d08ec5a00 category.572dd173939e250d08ec59ab category.55b6740de7798944b47924ec category.55b6740de7798944b47924ec category.57e51519805d8905e3830923 category.57e51519805d8905e3830923 category.55b6740de7798944b4792502 category.54f184c9b4c4d616d3cf2263 category.57bd8e7f805d896b0686cd75 category.55b6740de7798944b4792568 category.56a6f5b08e7989cd56fcea9d category.54f184dcb4c4d616d1cf226b category.55cdd55f830f780e86a03235 category.5590e712d48bac2951b9bbae category.55b6740de7798944b479256d category.55b6740de7798944b4792564 category.54f18577b4c4d616eccf22cc category.57bd8e7f805d896b0686cd73 category.54f18577b4c4d616eccf22cc category.54f18739b4c4d616d5cf2274 category.54f186e0b4c4d616d0cf228a category.55cdd55f830f780e86a031e4 category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc1a category.5590e713d48bac2951b9bc1a category.572dafad561feb5795a50e0c category.572dd171939e250d08ec5911 category.57bd8e7f805d896b0686cd73 category.56a6f5b08e7989cd56fcea9d category.56a6f5b08e7989cd56fcea9d category.56a6f5b08e7989cd56fcea9d category.54f184c9b4c4d616d3cf2263 category.5590e712d48bac2951b9bbbe category.5590e712d48bac2951b9bbf8 category.5590e712d48bac2951b9bbbe category.57bd8e7f805d896b0686cd75 category.55cdd55e830f780e86a0315b category.57bd8e7f805d896b0686cd75 category.5590e712d48bac2951b9bbe9 category.5590e712d48bac2951b9bbe9 category.54f18719b4c4d616d5cf2273 category.54f18739b4c4d616d5cf2274',
'mao': 'category.56a6f5b58e7989cd56fcea9e category.57bd8e7f805d896b0686cd73 category.54f186dcb4c4d616d0cf2289 category.572d9ebf561feb5795a50df9 category.57bd8e7f805d896b0686cd76 category.57e51518805d8905e38308e4 category.56a6f5b58e7989cd56fcea9e category.57bd8e7f805d896b0686cd74 category.57bd8e7f805d896b0686cd74 category.57e51518805d8905e38308e4 category.56a6f5b58e7989cd56fcea9e category.57e51518805d8905e38308e4 category.57e51518805d8905e38308e4 category.54f1853db4c4d616eccf22c9 category.55cdd55e830f780e86a0315b category.56a6f5b58e7989cd56fcea9e category.55b6740de7798944b4792534 category.55b6740de7798944b47924f5 category.572dafad561feb5795a50e0c category.55cdd55e830f780e86a03189 category.56a6d3048e7989cd56fce9e9 category.56a6f5b58e7989cd56fcea9e category.54f18577b4c4d616eccf22cc category.57e51518805d8905e38308e5 category.5590e712d48bac2951b9bbbe category.57bd8e7f805d896b0686cd73 category.5590e713d48bac2951b9bc14 category.56a6f5a28e7989cd56fcea9b category.5590e712d48bac2951b9bbbe category.55cdd55e830f780e86a0315b category.55cdd55f830f780e86a031b8 category.55cdd55f830f780e86a031cf category.572ddd66939e251de9f42f62 category.55cdd55f830f780e86a031b3 category.56a6f5a28e7989cd56fcea9b category.55cdd55e830f780e86a03192 category.5590e713d48bac2951b9bc19 category.5590e713d48bac2951b9bc19 category.56a6e8338e7989cd56fcea54 category.5590e713d48bac2951b9bc14 category.55cdd55e830f780e86a03135 category.54f18577b4c4d616eccf22cc category.54f186e0b4c4d616d0cf228a category.572dd173939e250d08ec59ee category.572dd173939e250d08ec59ee category.572dd171939e250d08ec58fe category.54f1868eb4c4d616eccf22db category.55b6740de7798944b4792564 category.56a6dc9f8e7989cd56fcea1c category.56a6e8cf8e7989cd56fcea58 category.573fe388939e257b45dc773f category.573fe388939e257b45dc7743 category.572dafe2561feb5795a50e0f category.56a6f6b78e7989cd56fceaa0 category.57bd8e7f805d896b0686cd74 category.56a6f5b58e7989cd56fcea9e category.5697041140a25b5bd66883d9 category.54f186dcb4c4d616d0cf2289 category.5590e712d48bac2951b9bbbf category.56a6e8cf8e7989cd56fcea58 category.56a6f5b58e7989cd56fcea9e category.57e51518805d8905e38308e4 category.54f1853db4c4d616eccf22c9 category.56a6e2258e7989cd56fcea31 category.572dd171939e250d08ec58b3 category.54f18719b4c4d616d5cf2273 category.57bd8e7f805d896b0686cd73 category.54f18719b4c4d616d5cf2273 category.54f18719b4c4d616d5cf2273 category.54f18719b4c4d616d5cf2273 category.56a6e1608e7989cd56fcea2b category.54f18719b4c4d616d5cf2273 category.5590e713d48bac2951b9bc14 category.5590e712d48bac2951b9bbbf category.56a6f5b58e7989cd56fcea9e category.54f186e0b4c4d616d0cf228a category.573fe388939e257b45dc773f category.573fe388939e257b45dc773f category.57bd8e7f805d896b0686cd76 category.5590e712d48bac2951b9bbe8 category.574962cc939e2554fa18494d category.5697011040a25b5bd66883cd category.572dd172939e250d08ec593e category.55b6740de7798944b4792545 category.55cdd55e830f780e86a03192 category.572dd171939e250d08ec58b3 category.56a6f66a8e7989cd56fcea9f category.573fe388939e257b45dc773f category.573fe388939e257b45dc773f category.573fe386939e257b45dc7732 category.5697047f40a25b5bd66883da category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.57bd8e7f805d896b0686cd77 category.5590e712d48bac2951b9bbbe category.55b6740de7798944b47924fc category.57e51518805d8905e38308e4 category.56a6e20b8e7989cd56fcea2f category.56937c613a22e443b2330758 category.54f184dcb4c4d616d1cf226b category.57e51519805d8905e38308f6 category.572dd171939e250d08ec58fe category.57bd8e7f805d896b0686cd77 category.55cdd55f830f780e86a031a4 category.57e51519805d8905e38308f5 category.56a6f5b58e7989cd56fcea9e category.54f186e0b4c4d616d0cf228a category.54f186e0b4c4d616d0cf228a category.5590e713d48bac2951b9bc14 category.569703c640a25b5bd66883d8 category.55cdd55f830f780e86a031c7 category.572dd173939e250d08ec59ee category.54f18719b4c4d616d5cf2273 category.5590e712d48bac2951b9bbd6 category.572dd173939e250d08ec59b5 category.5590e713d48bac2951b9bc14 category.55cdd55f830f780e86a031c8 category.54f18719b4c4d616d5cf2273 category.572dd173939e250d08ec5a00 category.574962cc939e2554fa184946 category.574962cc939e2554fa184932 category.56a6e5438e7989cd56fcea45 category.58ca004a805d891d72d2ef4d category.57bd8e7f805d896b0686cd77 category.55b6740de7798944b479249e category.58ca004f805d891d72d2ef54 category.572dd172939e250d08ec599b category.572dd171939e250d08ec58b0 category.573fe387939e257b45dc7736 category.54f19ce1b4c4d660fcf18bb2 category.572dd172939e250d08ec5985 category.572dd172939e250d08ec5985 category.572dd172939e250d08ec5985 category.572dd171939e250d08ec58b4 category.54f18e8ab4c4d6548376b720 category.572dd172939e250d08ec5968 category.55b6740de7798944b4792502 category.54f18577b4c4d616eccf22cc category.572dd171939e250d08ec58b3 category.57bd8e7c805d896b0686cd5b category.582ee9d7805d8911cdd1d6e8 category.54f198b6b4c4d660fcf18b7e category.55b6740de7798944b47924fa category.54f186e0b4c4d616d0cf228a category.56937b213a22e443b233074e category.56a6e5e08e7989cd56fcea47 category.572dd171939e250d08ec58b3 category.572dd172939e250d08ec593e category.572dd172939e250d08ec593e category.572dd171939e250d08ec58b3 category.572dd171939e250d08ec58b3 category.572dd171939e250d08ec58b3 category.572dd173939e250d08ec59ae category.572dd171939e250d08ec58b3 category.5697024340a25b5bd66883d3 category.569703c640a25b5bd66883d8 category.55b6740de7798944b4792573 category.55b6740de7798944b479256d category.55cdd55f830f780e86a031ae category.55cdd55e830f780e86a0315b category.5590e712d48bac2951b9bbbe category.55b6740de7798944b47924f6 category.56a6f5b58e7989cd56fcea9e category.54f1868eb4c4d616eccf22db category.55b6740de7798944b47924a0 category.57e51518805d8905e38308e4 category.56a6f5b58e7989cd56fcea9e category.5590e712d48bac2951b9bbbe category.54f18577b4c4d616eccf22cc category.5697024340a25b5bd66883d3 category.55cdd55f830f780e86a031b4 category.56a6e52b8e7989cd56fcea44 category.57bd8e7f805d896b0686cd73 category.56a6dc308e7989cd56fcea16 category.572dd171939e250d08ec58b3 category.56a6e3468e7989cd56fcea36 category.573fe387939e257b45dc7735 category.55b6740de7798944b4792505 category.56a6e3468e7989cd56fcea36 category.54f186e0b4c4d616d0cf228a category.56a6e3468e7989cd56fcea36 category.56a6e3468e7989cd56fcea36 category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.5697089c40a25b5bd66883e4 category.5590e712d48bac2951b9bbe2 category.55b6740de7798944b4792501 category.55b6740de7798944b4792501 category.56a6dc1a8e7989cd56fcea14 category.56a6dc1a8e7989cd56fcea14 category.573bcd79939e254b1b169b6a category.55b6740de7798944b47924fb category.55b6740de7798944b4792571 category.574962cc939e2554fa18495d category.574962cc939e2554fa18495d category.55cdd55e830f780e86a03180 category.57bd8e7f805d896b0686cd73 category.572dd171939e250d08ec5915 category.55b6740de7798944b4792521 category.56a6e5e08e7989cd56fcea47 category.56a6e20b8e7989cd56fcea2f category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.55b6740de7798944b4792573 category.55cdd55e830f780e86a0315b category.5590e712d48bac2951b9bbe2 category.56a6e5e08e7989cd56fcea47 category.573fe385939e257b45dc7730 category.55b6740de7798944b47924d1 category.55b6740de7798944b47924d1 category.572dd171939e250d08ec589d category.572dd171939e250d08ec589d category.572dd172939e250d08ec598b category.572dd171939e250d08ec5936 category.57e51518805d8905e38308e4 category.573fe387939e257b45dc7735 category.5590e712d48bac2951b9bbbe category.57bd8e7f805d896b0686cd73 category.54f1868eb4c4d616eccf22db category.54f1868eb4c4d616eccf22db category.56a6f5b58e7989cd56fcea9e category.582d4db9805d89419841331b category.55b6740de7798944b4792573 category.572dd171939e250d08ec58a5 category.56a6dc1a8e7989cd56fcea14 category.56a6dc1a8e7989cd56fcea14 category.55b6740de7798944b479249e category.55b6740de7798944b479249e category.56a6dc308e7989cd56fcea16 category.56a6e3468e7989cd56fcea36 category.56a6dc1a8e7989cd56fcea14 category.56a6dc1a8e7989cd56fcea14 category.55b6740de7798944b47924d1 category.572dd173939e250d08ec59ae category.572dd171939e250d08ec58b3 category.573fe385939e257b45dc7730 category.56a6f5b58e7989cd56fcea9e category.56a6f5b58e7989cd56fcea9e category.56a6e20b8e7989cd56fcea2f category.57832491939e2577cc79e7e5 category.55b6740de7798944b479256d category.54f1853db4c4d616eccf22c9 category.572dd172939e250d08ec5941 category.572dd172939e250d08ec5941 category.5590e712d48bac2951b9bbbe category.572dd173939e250d08ec59cf category.55b6740de7798944b4792573 category.55b6740de7798944b4792573 category.54f18a0fb4c4d616eccf2318 category.55b6740de7798944b479257e category.55cdd55e830f780e86a0312f category.55cdd55e830f780e86a03180 category.55cdd55e830f780e86a03180 category.55cdd55e830f780e86a03180 category.57bd8e7f805d896b0686cd77 category.56a6f66a8e7989cd56fcea9f category.58ca004a805d891d72d2ef4d category.56a6e20b8e7989cd56fcea2f category.5590e713d48bac2951b9bc19 category.54f18a0fb4c4d616eccf2318 category.57e51518805d8905e38308e8 category.5697047f40a25b5bd66883da category.55cdd55e830f780e86a0317f category.54f19ce1b4c4d660fcf18bb2 category.55b6740de7798944b4792520 category.55b6740de7798944b4792571 category.58ca004d805d891d72d2ef50 category.572dd173939e250d08ec59f3 category.58ed9949805d897fb153e0d6 category.55cdd55f830f780e86a0323b category.572da811561feb5795a50e09 category.5590e712d48bac2951b9bbc5 category.572dd172939e250d08ec594a category.57bd8e7f805d896b0686cd77 category.57bd8e7f805d896b0686cd75 category.572dd173939e250d08ec59fa category.572dd173939e250d08ec59fa category.572dd172939e250d08ec594a category.57bd8e7f805d896b0686cd77 category.57bd8e7f805d896b0686cd75 category.572dd173939e250d08ec59fa category.572dd173939e250d08ec59fa category.572dd171939e250d08ec58b3 category.572dd172939e250d08ec5985 category.572dd172939e250d08ec5985 category.572dd172939e250d08ec5993 category.56a6e20b8e7989cd56fcea2f category.56a6f2538e7989cd56fcea90 category.55cdd55e830f780e86a03130 category.574962cc939e2554fa184960 category.572dd171939e250d08ec58b3 category.56970b4240a25b5bd66883e8 category.574962cc939e2554fa184952 category.5590e713d48bac2951b9bc14 category.56a6f5a28e7989cd56fcea9b category.55cdd55e830f780e86a03192 category.56a6f5b58e7989cd56fcea9e category.5590e712d48bac2951b9bbbf category.56a6f5b58e7989cd56fcea9e category.55b6740de7798944b47924e0 category.56a6f5a28e7989cd56fcea9b category.5590e713d48bac2951b9bc14 category.5590e713d48bac2951b9bc14 category.56a6e20b8e7989cd56fcea2f category.54f1853db4c4d616eccf22c9 category.5590e712d48bac2951b9bbbe category.56a6e20b8e7989cd56fcea2f category.54f186e0b4c4d616d0cf228a category.56a6d3048e7989cd56fce9e9 category.56a6f2538e7989cd56fcea90 category.55cdd55e830f780e86a0315b category.56a6e20b8e7989cd56fcea2f',
'tiger': 'category.572dd171939e250d08ec5909 category.55b6740de7798944b4792564 category.56a6e2258e7989cd56fcea31 category.572dd171939e250d08ec58f4 category.572dd171939e250d08ec58a5 category.572da811561feb5795a50e09 category.5590e712d48bac2951b9bbbe category.55cdd55e830f780e86a0313a category.572dd172939e250d08ec598d category.5697023540a25b5bd66883d2 category.54f185b4b4c4d616eecf22a8 category.54f1853db4c4d616eccf22c9 category.55b6740de7798944b4792552 category.572dd172939e250d08ec5987 category.55cdd55f830f780e86a03242 category.57849731939e25055cd72091 category.5590e713d48bac2951b9bc14 category.56a6e2258e7989cd56fcea31 category.58ca004a805d891d72d2ef4d category.5590e713d48bac2951b9bc19 category.57bd8e7e805d896b0686cd67 category.572dd173939e250d08ec59f7 category.572dd172939e250d08ec5955 category.572dd173939e250d08ec59e7 category.56a6e2578e7989cd56fcea33 category.57bd8e7e805d896b0686cd67 category.572dd173939e250d08ec59f7 category.572dd173939e250d08ec59e7 category.56a6e2578e7989cd56fcea33 category.5590e713d48bac2951b9bc19 category.5697023540a25b5bd66883d2 category.5590e713d48bac2951b9bc19 category.55cdd55e830f780e86a03192 category.57bd8e7f805d896b0686cd77 category.5697023540a25b5bd66883d2 category.54f185b4b4c4d616eecf22a8 category.572dd172939e250d08ec5955 category.572dd172939e250d08ec5955 category.57bd8e7f805d896b0686cd77 category.569701c240a25b5bd66883d0 category.56a6f5b58e7989cd56fcea9e category.57bd8e7f805d896b0686cd73 category.572dd173939e250d08ec59a1 category.56a6dc0e8e7989cd56fcea13 category.56a6dc0e8e7989cd56fcea13 category.55cdd55e830f780e86a03192 category.574962cc939e2554fa184959 category.54f1853db4c4d616eccf22c9 category.56a6f24f8e7989cd56fcea8f category.574962cd939e2554fa18496f category.572dd171939e250d08ec58b3 category.54f184dcb4c4d616d1cf226b category.5697011040a25b5bd66883cd category.572dd173939e250d08ec59a3 category.57e51519805d8905e3830921 category.55b6740de7798944b4792509 category.56a6dc1a8e7989cd56fcea14 category.56a6e3468e7989cd56fcea36 category.56a6e3468e7989cd56fcea36 category.56a6e3468e7989cd56fcea36 category.54f184dcb4c4d616d1cf226b category.55b6740de7798944b4792573 category.54f184dcb4c4d616d1cf226b category.54f184dcb4c4d616d1cf226b category.5590e713d48bac2951b9bc14 category.5784972f939e25055cd72090 category.57bd8e7e805d896b0686cd6b category.572dd171939e250d08ec58f2 category.57bd8e7e805d896b0686cd65 category.572dd172939e250d08ec5941 category.572dd172939e250d08ec5941 category.55b6740de7798944b47924b5 category.572dd174939e250d08ec5a06 category.5590e712d48bac2951b9bbb3 category.572dd171939e250d08ec58cd category.57e51519805d8905e38308f6 category.56a6f5b58e7989cd56fcea9e category.56a6f5a28e7989cd56fcea9b category.572dd173939e250d08ec59a3 category.572dd173939e250d08ec59a3 category.56a6dc3e8e7989cd56fcea17 category.56a6dc8c8e7989cd56fcea1b category.56a6dc0e8e7989cd56fcea13 category.56a6dc3e8e7989cd56fcea17 category.54f1908db4c4d660f7f18aa5 category.54f1908db4c4d660f7f18aa5 category.5590e712d48bac2951b9bbbe category.54f190a8b4c4d660fcf18aa5 category.5590e712d48bac2951b9bbbe category.572dd173939e250d08ec59ab category.57bd8e7e805d896b0686cd65 category.5784972f939e25055cd72090 category.58ed9946805d897fb153dff5 category.58ed9944805d897fb153df8b category.55cdd55f830f780e86a031f8 category.56a6e74a8e7989cd56fcea50 category.5731c8bf939e254948159bbd category.55cdd55e830f780e86a03167 category.572dd173939e250d08ec59ab category.55cdd55f830f780e86a031f6',
'siqi': 'category.572da91d561feb5795a50e0a category.572dafd2561feb5795a50e0e category.56a6f5b08e7989cd56fcea9d category.56a6f5b58e7989cd56fcea9e category.54f1853db4c4d616eccf22c9 category.54f18953b4c4d616eecf2$cc category.56a6d3048e7989cd56fce9e9 category.5697041140a25b5bd66883d9 category.5590e713d48bac2951b9bc19 category.55cdd55e830f780e86a0315b category.55cdd55f830f780e86a031b0 category.54f18957b4c4d616eecf22cd cate$ory.55b6740de7798944b4792573 category.55b6740de7798944b4792573 category.55cdd55e830f780e86a0315b category.54f1853db4c4d616eccf22c9 category.57832491939e2577cc79e7e5 category.54f184c9b4c4d616d3cf2263 category.578$2491939e2577cc79e7e5 category.57832491939e2577cc79e7e5 category.54f184c9b4c4d616d3cf2263 category.57832491939e2577cc79e7e5 category.572dd171939e250d08ec58a5 category.5590e713d48bac2951b9bc0b category.55cdd55e830$780e86a0313a category.54f189b9b4c4d616d0cf2296',
'runchao': 'category.572dd172939e250d08ec593c category.54f186e0b4c4d616d0cf228a category.54f186e0b4c4d616d0cf228a category.572dd171939e250d08ec58ba category.574962cd939e2554fa184965 category.56a6e2258e7989cd56fce$31 category.572dd173939e250d08ec59d5 category.57e51518805d8905e38308f0 category.572dd172939e250d08ec5956 category.55cdd55e830f780e86a0315c category.572ddd66939e251de9f42f62 category.55b6740de7798944b4792573 cate$ory.57bd8e7a805d896b0686cd3e category.57e51519805d8905e3830922 category.55b6740de7798944b4792573 category.5590e713d48bac2951b9bc14 category.55b6740de7798944b4792573 category.55b6740de7798944b4792564 category.55b$740de7798944b4792573 category.5590e712d48bac2951b9bbae category.55b6740de7798944b479256d category.55b6740de7798944b479256d category.56a6e84c8e7989cd56fcea56 category.56a6e74a8e7989cd56fcea50 category.56a6e7698e7$89cd56fcea52 category.55cdd55f830f780e86a03201 category.5590e712d48bac2951b9bbae category.5590e712d48bac2951b9bbae category.5590e712d48bac2951b9bbae category.572dd171939e250d08ec58b3 category.57e51518805d8905e38$08e4 category.56a6e82c8e7989cd56fcea53 category.56a6e82c8e7989cd56fcea53 category.55cdd55e830f780e86a03167 category.5590e712d48bac2951b9bbbe category.55b6740de7798944b4792564 category.54f1853db4c4d616eccf22c9 ca$egory.55b6740de7798944b4792573 category.55cdd55e830f780e86a03185 category.55b6740de7798944b479256d category.55cdd55e830f780e86a03185 category.5590e712d48bac2951b9bbbe category.56a6e74a8e7989cd56fcea50 category.5$cdd55e830f780e86a03185 category.55b6740de7798944b479256d category.55cdd55e830f780e86a03185 category.55b6740de7798944b4792564 category.55b6740de7798944b4792564 category.55b6740de7798944b4792564 category.55cdd55e8$0f780e86a03185 category.55b6740de7798944b4792564 category.572dd172939e250d08ec594a category.56a6de2f8e7989cd56fcea22 category.54f184dcb4c4d616d1cf226b category.54f19623b4c4d660e6f18a96 category.572dd171939e250d0$ec58bf category.572dd171939e250d08ec58bf category.56a6eabe8e7989cd56fcea61 category.54f1853db4c4d616eccf22c9 category.54f184dcb4c4d616d1cf226b category.54f186dcb4c4d616d0cf2289 category.56a6eabe8e7989cd56fcea61 $ategory.54f186e0b4c4d616d0cf228a category.55b6740de7798944b4792573 category.5590e713d48bac2951b9bc14 category.5590e713d48bac2951b9bc14 category.572dd171939e250d08ec58bf category.55b6740de7798944b479252d',
'roger': 'category.56a6e8338e7989cd56fcea54 category.55cdd55e830f780e86a0315b category.55cdd55e830f780e86a0315b category.55cdd55f830f780e86a031a4 category.55b6740de7798944b4792559 category.55b6740de7798944b4792559 category.55b6740de7798944b4792559 category.55b6740de7798944b4792559 category.55b6740de7798944b4792559 category.57e51519805d8905e38308fa category.54f190a4b4c4d660f1f18a9b'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=200,
                       help='number of words to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                       help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                       help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def sample(args):
    tag_id_name_dict = dict()
    tag_name_id_dict = dict()
    for line in open('/home/icarus/yhu/category_name_dict/data'):
        cat_id, name = line.split('|')[:2]
        tag_id_name_dict['category.' + cat_id] = name
        tag_name_id_dict[name] = 'category.' + cat_id
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for user, serials in sample_user.items():
                print(user)
                print(model.sample(sess, words, vocab, args.n, serials, args.sample, args.pick, args.width, tag_id_name_dict))

if __name__ == '__main__':
    main()


