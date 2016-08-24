# Rovina Semantic Segmentation
This repo contains the ros package for semantic segmentation of local maps within the context of the ROVINA project.

You will need the dataset which is not included here and you will need the fps mapper from UNIROMA as this package runs on top of the maps created by that mapper.

For questions feel free to contact me at hermans@vision.rwth-aachen.de

# Running it

```bash
$ roslaunch semantic_segmentation semantics.launch
```
There are three services:

- `/semantic_segmentation/local_map_ids`
  returns a list of all local maps a segmentation is available for.
- `/semantic_segmentation/information`
  returns all the information about the different layers available, the color codings and the class names.
- `/semantic_segmentation/get_local_map_segmentation`
  provides the semantic segmentation results for a requested local map id. This can consist of one or more layers of semantics, e.g. a *material* and an *object* layer.

The semantics use a random forest on the key frames and a dense crf to smooth the global predictions in a local map. Parameters can be set in the config file. You will need to download the trained model with:

```bash
$ roscd semantic_segmentation
$ cd resources
$ ./get_rf_model.sh
```

Whenever you update the code, also download a new model, or if you download a new model, also update the code. The configuration of the classes and the used features during training is stored in a config file in the repo. Mismatches between the model and the config will result in segfaults.

Visualization can be done using the Mission Control Interface developed by Algorithmica.

# Issues
The code is far from polished and it is not meant as a ready to use semantic segmentation package for ROS. It was explicitly developed in the context of the [ROVINA project](http://www.rovina-project.eu). Running it directly will likely not work out. You can however reuse parts and ideas in this repository.

# Licensing
For ease of use this repo contains code from several other places in the third-party folder:
* Libforest is a random forest library developed by a friend of mine who will hopefully at some point get around to releasing it seriously. The version included is heavily edited to support multiple label predictions in a single decision tree. The original project can be found [here](https://github.com/TobyPDE/libforest).
* JsonCpp library, which is under the MIT license.
* Dense CRF from Philipp Krähenbühl and Vladlen Koltun. Get your original [here](http://www.philkr.net/home/densecrf).

This project uses the MIT License.
