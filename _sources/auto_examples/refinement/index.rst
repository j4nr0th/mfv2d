

.. _sphx_glr_auto_examples_refinement:

.. _mfv2d.examples.refinement:

Refinemet Examples
==================

Examples of the code.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One of key features of mfv2d is the ability to locally refine the mesh with both hierarchical refinement (divide elements) or with polynomial refinement (increase order of elements).">

.. only:: html

  .. image:: /auto_examples/refinement/images/thumb/sphx_glr_plot_direct_poisson_refined_pre_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_refinement_plot_direct_poisson_refined_pre.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Poisson Equation with Local Refinement</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="As mentioned in sphx_glr_auto_examples_refinement_plot_direct_poisson_refined_pre.py example, refinement can be done &quot;post-solver&quot; as well. This means that mesh refinement is performed after the solver finishes running. This does not change the computed solution, but allows for next solve to be more accurate, if repeated.">

.. only:: html

  .. image:: /auto_examples/refinement/images/thumb/sphx_glr_plot_direct_poisson_refined_post_p_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_refinement_plot_direct_poisson_refined_post_p.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-Solver p-Refinement</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This solver showcases refinement process using post-solver refinement. The idea here is to use both h-refinement and p-refinement, which allows for much more efficient convergence for locally sharp solutions.">

.. only:: html

  .. image:: /auto_examples/refinement/images/thumb/sphx_glr_plot_direct_poison_post_hp_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_refinement_plot_direct_poison_post_hp.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-Solver Combined hp-refinement with Direct Poisson</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Using exact solution as a refinement criterion is a luxury that&#x27;s not often availabel for real problems. As such, process in the example sphx_glr_auto_examples_refinement_plot_direct_poison_post_hp.py may not be possible to replicate.">

.. only:: html

  .. image:: /auto_examples/refinement/images/thumb/sphx_glr_plot_advdif_post_hp_projection_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_refinement_plot_advdif_post_hp_projection.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Approximating Error from Projections with Advection-Diffusion</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/refinement/plot_direct_poisson_refined_pre
   /auto_examples/refinement/plot_direct_poisson_refined_post_p
   /auto_examples/refinement/plot_direct_poison_post_hp
   /auto_examples/refinement/plot_advdif_post_hp_projection

