#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import nsNLP
import numpy as np
from tools.functions import create_tokenizer
import os


####################################################
# Functions
####################################################


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


# Create directory
def create_directories(output_directory, xp_name):
    """
    Create image directory
    :return:
    """
    # Directories
    image_directory = os.path.join(output_directory, xp_name, "images")
    texts_directory = os.path.join(output_directory, xp_name, "texts")

    # Create if does not exists
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)
    # end if

    # Create if does not exists
    if not os.path.exists(texts_directory):
        os.mkdir(texts_directory)
    # end if

    return image_directory, texts_directory
# end create_directories


# Save embedding images
def save_embedding_images():
    """
    Save embedding images
    :param text_vectors:
    :param size:
    :param desc_info:
    :return:
    """
    # Embeddings
    embeddings = nsNLP.embeddings.Embeddings((reservoir_size*state_gram)+1)

    # For each vectors
    for text in text_embeddings.keys():
        embeddings.add(text, text_embeddings[text])
        embeddings.set(text, 'count', 1)
    # end for

    # Export image of reduced vectors with TSNE
    embeddings.wordnet('count',
                       os.path.join(image_directory, u"wordnet_TSNE_" + unicode(w_index) + u".png"),
                       n_words=args.n_authors*100,
                       fig_size=args.fig_size, reduction='TSNE', info=desc_info)

    # Export image of reduced vectors with PCA
    embeddings.wordnet('count',
                       os.path.join(image_directory, u"wordnet_PCA_" + unicode(w_index) + u".png"),
                       n_words=args.n_authors*100,
                       fig_size=args.fig_size, reduction='PCA', info=desc_info)
# end save_embeddings_images

####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":

    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)

    # Author parameters
    args.add_argument(command="--n-authors", name="n_authors", type=int,
                      help="Number of authors to include in the test", default=15, extended=False)
    for i in range(15):
        args.add_argument(command="--author{}".format(i), name="author{}".format(i), type=str,
                          help="{}th author to test".format(i), extended=False)
    # end for

    # ESN arguments
    args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                      required=True, extended=True)
    args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                      default="1.0", extended=True)
    args.add_argument(command="--leak-rate", name="leak_rate", type=str, help="Reservoir's leak rate", extended=True,
                      default="1.0")
    args.add_argument(command="--input-scaling", name="input_scaling", type=str, help="Input scaling", extended=True,
                      default="0.5")
    args.add_argument(command="--input-sparsity", name="input_sparsity", type=str, help="Input sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--w-sparsity", name="w_sparsity", type=str, help="W sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--converters", name="converters", type=str,
                      help="The text converters to use (fw, pos, tag, wv, oh)", default='oh', extended=True)
    args.add_argument(command="--pca-path", name="pca_path", type=str, help="PCA model to load", default=None,
                      extended=False)
    args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                      extended=False)
    args.add_argument(command="--aggregation", name="aggregation", type=str, help="Output aggregation method",
                      extended=True, default="average")
    args.add_argument(command="--state-gram", name="state_gram", type=str, help="State-gram value",
                      extended=True, default="1")

    # Tokenizer and clustering parameters
    args.add_argument(command="--distance", name="distance", type=str, help="Distance measure to use", default='cosine',
                      extended=True)
    args.add_argument(command="--tokenizer", name="tokenizer", type=str,
                      help="Which tokenizer to use (spacy, nltk, spacy-tokens)", default='nltk', extended=False)
    args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters", default='en',
                      extended=False)

    # Experiment output parameters
    args.add_argument(command="--fig-size", name="fig_size", help="Figure size (pixels)", type=float, default=1024.0,
                      extended=False)
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--eval", name="eval", type=str, help="Evaluation measure", default='bsquared',
                      extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Create image directory
    image_directory, texts_directory = create_directories(args.output, args.name)

    # Corpus
    reteursC50 = nsNLP.data.Corpus(args.dataset)

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager\
    (
        args.output,
        args.name,
        args.description,
        args.get_space(),
        args.n_samples,
        1,
        verbose=args.verbose
    )

    # Author list
    authors = reteursC50.get_authors()[:args.n_authors]
    author_list = reteursC50.get_authors_list()[:args.n_authors]

    # Get text list
    texts = list()
    text_list = list()
    for author in authors:
        for text in author.get_texts():
            texts.append(text)
            text_list.append(text.get_title())
        # end for
    # end for

    # Print authors
    xp.write(u"Authors : {}".format(author_list), log_level=0)
    xp.write(u"Texts : {}".format(text_list), log_level=0)

    # First params
    rc_size = int(args.get_space()['reservoir_size'][0])
    rc_w_sparsity = args.get_space()['w_sparsity'][0]

    # Create W matrix
    w = nsNLP.esn_models.ESNTextClassifier.w(rc_size=rc_size, rc_w_sparsity=rc_w_sparsity)

    # W index
    w_index = 0

    # Last space
    last_space = dict()

    # Iterate
    for space in param_space:
        # Params
        reservoir_size = int(space['reservoir_size'])
        w_sparsity = space['w_sparsity']
        leak_rate = space['leak_rate']
        input_scaling = space['input_scaling']
        input_sparsity = space['input_sparsity']
        spectral_radius = space['spectral_radius']
        converter_desc = space['converters']
        aggregation = space['aggregation'][0][0]
        state_gram = space['state_gram']

        # Choose the right tokenizer
        if converter_in(converter_desc, "wv") or \
                converter_in(converter_desc, "pos") or \
                converter_in(converter_desc, "tag"):
            tokenizer = create_tokenizer("spacy_wv")
        else:
            tokenizer = create_tokenizer("nltk")
        # end if

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.n_samples):
            # Set sample
            xp.set_sample_state(n)

            # Description
            desc_info = u"{}-{}".format(space, n)

            # Create ESN text classifier
            classifier = nsNLP.esn_models.ESNTextClassifier.create\
            (
                classes=text_list,
                rc_size=reservoir_size,
                rc_spectral_radius=spectral_radius,
                rc_leak_rate=leak_rate,
                rc_input_scaling=input_scaling,
                rc_input_sparsity=input_sparsity,
                rc_w_sparsity=w_sparsity,
                converters_desc=converter_desc,
                use_sparse_matrix=True if converter_in(converter_desc, "oh") else False,
                w=w if args.keep_w else None,
                aggregation=aggregation,
                state_gram=state_gram
            )

            # Save w matrix
            if not args.keep_w:
                xp.save_object(u"w_{}".format(w_index), classifier.get_w(), info=u"{}".format(space))
            # end if

            # Add texts
            for index, text in enumerate(texts):
                # Add
                classifier.train(tokenizer(text.x()), text.get_title())
            # end for

            # Train
            classifier.finalize(verbose=False)

            # Extract text embeddings
            text_embeddings = classifier.get_embeddings()

            # Save embedding images
            save_embedding_images()

            # Clustering
            clustering = nsNLP.clustering.Clutering()

            # Add each text
            for text_title in text_list:
                clustering.add(reteursC50.get_by_title(text_title), text_embeddings[text_title])
            # end for

            # Compute K-mean clusters, and save Bcubed
            clusters = clustering.k_means(k=args.n_authors)
            result = nsNLP.measures.Clustering().bcubed_f1(clusters)
            xp.add_result(result)

            # Compute hierarchical clusters, and save dendrogram
            linkage_matrix = clustering.hierarchical_clustering(os.path.join(image_directory, u"dendrogram_" + unicode(w_index) + u".png"))

            # Delete classifier
            del classifier
            del clustering

            # W index
            w_index += 1
        # end for samples

        # Last space
        last_space = space
    # end for

    # Save experiment results
    xp.save()
# end if
