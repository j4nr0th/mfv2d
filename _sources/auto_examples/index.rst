:orphan:

.. _mfv2d.examples:

Examples
========

Examples of the code.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


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


Examples Unsteady
=================

Examples of the code.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cavity flow can be considered a canonical solution for the Navier-Stokes, given how well the solution to this problem is known. In this exampled it is solved for the case of Re = 10, since that allows for quick convergence on a fairly coarse grid.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_cavity_flow_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_cavity_flow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Navier-Stokes: Cavity Flow</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The (unsteady) heat equation is essentially just the Poisson equation with a time derivative term added to the left side, as per equation unsteady-heat-direct-equation.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_heat_direct_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_heat_direct.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Unsteady Heat Equations in Direct Formulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Vector reaction equation examples is solving the same equations as sphx_glr_auto_examples_unsteady_plot_reaction.py and sphx_glr_auto_examples_unsteady_plot_reaction_mixed.py, but with the u being a 1-form. This does just makes the solution have two decoupled components, which are solved for independently.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_vector_reaction_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_vector_reaction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Vector Reaction Equation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example is exactly the same as sphx_glr_auto_examples_unsteady_plot_heat_direct.py. As such, only differences from that one will be mentioned.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_heat_mixed_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_heat_mixed.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Unsteady Heat Equations in Mixed Formulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Mixed reaction equation examples is exactly the same equation as the one in sphx_glr_auto_examples_unsteady_plot_reaction.py, but with the difference being that now u is taken to be a 2-form instead of a 0-form. As such, there is now a real necessety to introduce a 1-form for its derivative.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_reaction_mixed_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_reaction_mixed.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mixed Reaction Equation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The reaction equation is probably the simples non-trivial unsteady differenital equation. It is given by equation unsteady-reaction-direct-equation. If the forcing f is only space dependent, then it has a very simple analytical solution given by unsteady-reaction-direct-solution, where u_\mathrm{initial} = u(x, y, 0) and u_\mathrm{final} = f(x, y) / \alpha.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_reaction_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_reaction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reaction Equation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/steady/index.rst
   /auto_examples/unsteady/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
