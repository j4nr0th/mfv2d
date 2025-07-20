

.. _sphx_glr_auto_examples_steady:

.. _mfv2d.examples.steady:

Steady Examples
===============

Examples of the code.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Incompressible Navier-Stokes is at the heart of modeling low-speed aerodynamics it can be seen as a Stokes flow eqution, with a non-linear source term \omega \times u. The full system is given as per as system steady-ns-equation. When written with differential geometry, it becomes system steady-stokes-diff-geom.">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_navier_stokes_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_navier_stokes.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Incompressible Navier-Stokes Equation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how the Poisson equation can be solved using the mixed formulation. This means that the equation steady-mixed-poisson-equation is formulated in two steps, given in equations steady-mixed-poisson-first and steady-mixed-poisson-second.">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_mixed_poisson_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_mixed_poisson.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Poisson Equation in the Mixed Formulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how linear advection-diffusion equation can be solved. There are of course two main ways:">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_linear_adv_dif_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_linear_adv_dif.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Linear Advection-Diffusion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how the Poisson equation can be solved using the direct formulation. This means that the equation steady-direct-poisson-equation is formulated directly without any intermediate steps.">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_direct_poisson_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_direct_poisson.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Poisson Equation in the Direct Formulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stokes flow is a steady state solution to the Poisson equation, with an added incompressibility constraint. It is also the symmetric part of Navier-Stokes equations, so solving it is the first step in solving Navier-Stokes.">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_stokes_flow_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_stokes_flow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stokes Flow</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One of key features of mfv2d is the ability to locally refine the mesh with both hierarchical refinement (divide elements) or with polynomial refinement (increase order of elements).">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_direct_poisson_refined_pre_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_direct_poisson_refined_pre.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Poisson Equation with Local Refinement</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="As mentioned in sphx_glr_auto_examples_steady_plot_direct_poisson_refined_pre.py example, refinement can be done &quot;post-solver&quot; as well. This means that mesh refinement is performed after the solver finishes running. This does not change the computed solution, but allows for next solve to be more accurate, if repeated.">

.. only:: html

  .. image:: /auto_examples/steady/images/thumb/sphx_glr_plot_direct_poisson_refined_post_p_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_steady_plot_direct_poisson_refined_post_p.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-Solver p-Refinement</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/steady/plot_navier_stokes
   /auto_examples/steady/plot_mixed_poisson
   /auto_examples/steady/plot_linear_adv_dif
   /auto_examples/steady/plot_direct_poisson
   /auto_examples/steady/plot_stokes_flow
   /auto_examples/steady/plot_direct_poisson_refined_pre
   /auto_examples/steady/plot_direct_poisson_refined_post_p

