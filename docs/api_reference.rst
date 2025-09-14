API Reference
=============

This section documents the public API for scXpand.

Main API
--------

The primary entry points for using scXpand:

.. automodule:: scxpand
   :members: run_inference, run_prediction_pipeline, download_pretrained_model, get_pretrained_model_info, list_pretrained_models, ModelType
   :show-inheritance:

Result Classes
--------------

Classes returned by the main API functions:

.. automodule:: scxpand.core.inference_results
   :members: InferenceResults, ModelInfo
   :show-inheritance:

Command Line Interface
----------------------

The CLI functions that power the ``scxpand`` command-line tool. These can also be called programmatically:

.. automodule:: scxpand.main
   :members: train, optimize, optimize_all, predict
   :show-inheritance:

.. note::
   These functions are primarily designed for CLI use. For programmatic use, prefer the main API functions above.

.. toctree::
   :maxdepth: 1

   api/data_util
   api/hyperopt
