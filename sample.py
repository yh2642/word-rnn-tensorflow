from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import json
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

sample_profile = '[{"572dd171939e250d08ec58c2":0.0625,"5590e712d48bac2951b9bbbc":0.3125,"54f186beb4c4d616eecf22b1":0.1875,"\u5305\u5305":0.5,"54f19584b4c4d660f7f18ae4":0.0625,"54f1853db4c4d616eccf22c9":0.375,"\u914d\u9970":0.5},{"\u5305\u5305":0.5,"\u914d\u9970":0.5},{"\u7a7f\u642d":0.2282201835,"\u978b\u5b50":0.1542995276,"\u5bb6\u5c45":0.1542995276,"\u5bb6\u88c5":0.1542995276,"\u5305\u5305":0.1544571069,"\u914d\u9970":0.1544241268},{"\u7a7f\u642d":0.2282201835,"\u978b\u5b50":0.1542995276,"\u9152\u6c34\u996e\u6599":1.0,"\u5305\u5305":0.1544571069,"\u5bb6\u5c45":0.1542995276,"\u914d\u9970":0.1544241268,"\u5bb6\u88c5":0.1542995276},{"\u62a4\u80a4\u5de5\u5177":0.0289877498,"\u9632\u6652":0.1123485665,"\u5305\u5305":0.1544571069,"\u62a4\u80a4":0.0564504259,"\u6e05\u6d01":0.1328260591,"\u5065\u8eab\u8fd0\u52a8":0.0066878522,"\u7a7f\u642d":0.2282201835,"\u914d\u9970":0.1544241268,"\u6297\u8001":0.0472084262,"\u5065\u8eab":0.2889508747,"\u8fd0\u52a8":0.0909855687,"\u9152\u6c34\u996e\u6599":1.0,"\u8fd0\u52a8\u88c5\u5907":0.0133757044,"\u5bb6\u5c45":0.1542995276,"\u5bb6\u88c5":0.1542995276,"\u4fdd\u6e7f":0.1169378939,"\u978b\u5b50":0.1542995276,"\u7cbe\u534e":0.0244623178,"\u9762\u971c":0.0807785608},{"\u8fd0\u52a8":0.0909855687,"\u62a4\u80a4\u5de5\u5177":0.0289877498,"\u9152\u6c34\u996e\u6599":1.0,"\u9632\u6652":0.1123485665,"\u62a4\u80a4":0.0564504259,"\u6e05\u6d01":0.1328260591,"\u9762\u971c":0.0807785608,"\u7cbe\u534e":0.0244623178,"\u5065\u8eab\u8fd0\u52a8":0.0066878522,"\u8fd0\u52a8\u88c5\u5907":0.0133757044,"\u6297\u8001":0.0472084262,"\u5065\u8eab":0.2889508747,"\u4fdd\u6e7f":0.1169378939},{"\u65c5\u884c":0.0186956634,"\u62a4\u80a4\u5de5\u5177":0.0289877498,"\u6297\u8001":0.0472084262,"\u5305\u5305":0.075887362,"\u62a4\u80a4":0.0564504259,"\u5bb6\u5c45":0.075887362,"\u6e05\u6d01":0.1328260591,"\u7f8e\u98df":0.0365296804,"\u5065\u8eab\u8fd0\u52a8":0.0066878522,"\u7a7f\u642d":0.1360424493,"\u914d\u9970":0.075887362,"\u8fd0\u52a8":0.2327484535,"\u98df\u54c1":0.0365296804,"\u5bb6\u88c5":0.075887362,"\u8fd0\u52a8\u88c5\u5907":0.0871206472,"\u5065\u8eab":0.2889508747,"\u4fdd\u6e7f":0.1169378939,"\u9632\u6652":0.1123485665,"\u98df\u8c31":0.1404109589,"\u978b\u5b50":0.0762046117,"\u7cbe\u534e":0.0244623178,"\u9910\u5385":0.0365296804,"\u9762\u971c":0.0807785608},{"\u65c5\u884c":0.0186956634,"\u5907\u5b55":0.0030598166,"\u5305\u5305":0.1695031112,"54f1868eb4c4d616eccf22db":0.2608695652,"\u7f8e\u98df":0.0365296804,"\u73a9\u5177":0.0112505914,"\u65b0\u751f\u513f":0.0030598166,"\u7a7f\u642d":0.1427016768,"\u914d\u9970":0.0825465894,"\u80b2\u513f\u77e5\u8bc6":0.0008289379,"\u8fd0\u52a8":0.1417628848,"\u98df\u54c1":0.0365296804,"\u5bb6\u5c45":0.0825465894,"\u8fd0\u52a8\u88c5\u5907":0.0737449428,"\u5bb6\u88c5":0.0825465894,"\u6bcd\u5a74":0.0104430949,"\u98df\u8c31":0.1404109589,"\u4e66\u7c4d":0.0665588156,"\u978b\u5b50":0.0828638391,"\u5b55\u5988":0.0152990831,"\u9910\u5385":0.0365296804,"54f19584b4c4d660f7f18ae4":0.0434782609,"\u65f6\u5c1a":0.0203977061,"54f1853db4c4d616eccf22c9":0.2608695652,"\u4ea7\u540e\u62a4\u7406":0.0030598166,"54f186afb4c4d616d0cf2286":0.1739130435},{"\u65c5\u884c":0.0186956634,"\u5907\u5b55":0.0030598166,"\u5305\u5305":0.1816053339,"54f1868eb4c4d616eccf22db":0.2608695652,"\u7f8e\u98df":0.0365296804,"\u73a9\u5177":0.0112505914,"\u6bcd\u5a74":0.0104430949,"\u7a7f\u642d":0.6500649037,"\u914d\u9970":0.0946488121,"\u80b2\u513f\u77e5\u8bc6":0.0008289379,"\u8fd0\u52a8":0.1417628848,"\u7537\u4eba":0.0047389957,"\u98df\u54c1":0.0365296804,"\u5bb6\u88c5":0.1183846404,"\u7537\u88c5":0.25,"\u8fd0\u52a8\u88c5\u5907":0.0737449428,"\u5bb6\u5c45":0.2424828449,"\u65b0\u751f\u513f":0.0030598166,"\u5bb6\u7528\u7535\u5668":0.0058168028,"\u98df\u8c31":0.1404109589,"\u4e66\u7c4d":0.0665588156,"\u978b\u5b50":0.0949660618,"\u5b55\u5988":0.0152990831,"\u9910\u5385":0.0365296804,"54f19584b4c4d660f7f18ae4":0.0434782609,"\u65f6\u5c1a":0.0203977061,"54f1853db4c4d616eccf22c9":0.2608695652,"\u4ea7\u540e\u62a4\u7406":0.0030598166,"54f186afb4c4d616d0cf2286":0.1739130435},{"\u5907\u5b55":0.0030598166,"\u5305\u5305":0.2872773715,"54f1868eb4c4d616eccf22db":0.2608695652,"\u73a9\u5177":0.0112505914,"\u6bcd\u5a74":0.0201432231,"\u7a7f\u642d":1.2372548943,"\u914d\u9970":0.0336541831,"\u80b2\u513f\u77e5\u8bc6":0.0008289379,"\u7537\u4eba":0.0047389957,"\u5bb6\u88c5":0.0573900114,"\u7537\u88c5":0.25,"\u5bb6\u5c45":0.1814882159,"\u65b0\u751f\u513f":0.0030598166,"\u5b9d\u5b9d\u5582\u517b":0.0051926047,"\u5bb6\u7528\u7535\u5668":0.0058168028,"\u4e66\u7c4d":0.0665588156,"\u978b\u5b50":0.0336541831,"\u5b55\u5988":0.030191816,"54f19584b4c4d660f7f18ae4":0.0434782609,"\u65f6\u5c1a":0.0203977061,"54f1853db4c4d616eccf22c9":0.2608695652,"\u4ea7\u540e\u62a4\u7406":0.0089115795,"54f186afb4c4d616d0cf2286":0.1739130435},{"\u7537\u88c5":0.25,"\u5bb6\u5c45":0.1748289884,"\u5b55\u5988":0.0148927329,"\u5305\u5305":0.1936616223,"\u5b9d\u5b9d\u5582\u517b":0.0051926047,"\u5bb6\u7528\u7535\u5668":0.0058168028,"\u4ea7\u540e\u62a4\u7406":0.0058517628,"\u6bcd\u5a74":0.0097001282,"\u7a7f\u642d":1.2305956669,"\u978b\u5b50":0.0269949556,"\u914d\u9970":0.0269949556,"\u7537\u4eba":0.0047389957,"\u5bb6\u88c5":0.0507307839},{"\u5bb6\u5c45":0.0148927329,"\u5bb6\u88c5":0.0148927329,"\u5305\u5305":0.1815593996,"\u5b9d\u5b9d\u5582\u517b":0.0051926047,"\u6bcd\u5a74":0.0097001282,"\u7a7f\u642d":0.7232324399,"\u978b\u5b50":0.0148927329,"\u5b55\u5988":0.0148927329,"\u4ea7\u540e\u62a4\u7406":0.0058517628,"\u914d\u9970":0.0148927329}]'

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
        tag_id_name_dict[cat_id] = name
        tag_name_id_dict[name] = cat_id
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
            print(model.sample(sess, words, vocab, args.n, json.loads(sample_profile), args.sample, args.pick, args.width, tag_id_name_dict))

if __name__ == '__main__':
    main()


