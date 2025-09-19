#!/bin/bash

echo '*** PATCH EXTRACTION PIPELINE ***'
./make_patch.sh

echo -e '\n*** PATCH ENCODING PIPELINE ***'
./make_embedding.sh

echo -e '\n*** CALCULATED MORPHOLOGY FEATURES ***'
./make_morphology.sh

echo -e '\n*** CONSTRUCTING GRAPHS ***'
./make_graph.sh

echo -e '\n*** TRAINING GNN ***'
./make_train.sh